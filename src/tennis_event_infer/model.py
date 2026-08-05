from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torchvision.models import resnet18


class _ConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.net(values)
        return self.norm((values + residual).transpose(1, 2)).transpose(1, 2)


class _TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, conv_layers: int, gru_layers: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("trajectory_hidden_dim 必须是正偶数")
        if conv_layers <= 0 or gru_layers <= 0:
            raise ValueError("轨迹卷积层数和 GRU 层数必须是正整数")
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Sequential(*[_ConvBlock(hidden_dim, dropout) for _ in range(conv_layers)])
        self.bigru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.embedding_dim = hidden_dim * 3

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        values = self.proj(trajectory)
        values = self.conv(values.transpose(1, 2)).transpose(1, 2)
        values, _ = self.bigru(values)
        return torch.cat(
            (values[:, values.shape[1] // 2], values.mean(dim=1), values.max(dim=1).values),
            dim=-1,
        )


class _ImageTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2:
            raise ValueError("hidden_dim 必须是正偶数")
        self.input_dim = input_dim
        self.embedding_dim = hidden_dim * 3
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.bigru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[2] != self.input_dim or values.shape[1] < 1:
            raise ValueError(f"patch embedding 必须为 [B,K,{self.input_dim}] 且 K>0")
        _validate_mask(mask, values)
        valid_samples = mask.any(dim=1)
        if not valid_samples.any():
            return values.new_zeros((values.shape[0], self.embedding_dim))
        projected = self.proj(values.masked_fill(~mask.unsqueeze(-1), 0))
        projected = projected.masked_fill(~mask.unsqueeze(-1), 0)
        samples = valid_samples.nonzero().flatten()
        sequences = [projected[index, mask[index]] for index in samples]
        lengths = torch.tensor([len(sequence) for sequence in sequences], device="cpu")
        packed = pack_padded_sequence(
            pad_sequence(sequences, batch_first=True), lengths, batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.bigru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)
        restored = projected.new_zeros((*projected.shape[:2], encoded.shape[-1]))
        mean_pool = projected.new_zeros((projected.shape[0], encoded.shape[-1]))
        max_pool = torch.zeros_like(mean_pool)
        for output_index, sample_index in enumerate(samples):
            sequence = encoded[output_index, : lengths[output_index]]
            restored[sample_index, mask[sample_index]] = sequence
            mean_pool[sample_index] = sequence.mean(dim=0)
            max_pool[sample_index] = sequence.max(dim=0).values
        pooled = torch.cat((restored[:, restored.shape[1] // 2], mean_pool, max_pool), dim=-1)
        return torch.where(valid_samples.unsqueeze(-1), pooled, torch.zeros_like(pooled))


class _PatchBranchEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.modalities = ("patch",)
        self.num_patch_qualities = 4
        self.continuous_quality = True
        self.full = None
        self.patch_quality_embedding = nn.Embedding(4, 8)
        self.patch_quality_continuous = nn.Sequential(nn.Linear(3, 8), nn.LayerNorm(8), nn.GELU())
        self.patch = _ImageTemporalEncoder(embedding_dim + 16, hidden_dim, dropout)
        self.branch_embedding_dim = hidden_dim * 3

    def forward(
        self,
        *,
        patch_image: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_quality: torch.Tensor,
        patch_quality_continuous: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        _validate_mask(patch_mask, patch_image)
        if patch_quality.dtype != torch.long or patch_quality.shape != patch_image.shape[:2]:
            raise ValueError("patch_quality 必须为 [B,K] long")
        if patch_quality.numel() and (patch_quality.min() < 0 or patch_quality.max() >= 4):
            raise ValueError("patch_quality 超出 0..3")
        if not torch.equal(patch_quality.eq(0), ~patch_mask):
            raise ValueError("patch_quality missing 状态必须与 patch_mask 一致")
        if patch_quality_continuous.shape != (*patch_image.shape[:2], 3):
            raise ValueError("patch_quality_continuous 必须为 [B,K,3]")
        if not patch_quality_continuous.is_floating_point() or not torch.isfinite(patch_quality_continuous).all():
            raise ValueError("patch_quality_continuous 必须是有限浮点数")
        values = torch.cat(
            (
                patch_image,
                self.patch_quality_embedding(patch_quality),
                self.patch_quality_continuous(patch_quality_continuous),
            ),
            dim=-1,
        )
        return [self.patch(values, patch_mask)], [patch_mask.float().mean(dim=1, keepdim=True)]


class _ResNet18Layer4Encoder(nn.Module):
    embedding_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.requires_grad_(False)
        for module in self.backbone.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                continue
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.layer4.train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pixels.dtype != torch.uint8 or pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("patch_pixels 必须为 [B,K,3,H,W] uint8")
        _validate_mask(mask, pixels)
        batch_size, steps = pixels.shape[:2]
        output = torch.zeros(batch_size * steps, self.embedding_dim, device=pixels.device)
        valid = mask.reshape(-1)
        if not valid.any():
            return output.reshape(batch_size, steps, self.embedding_dim)
        values = pixels.reshape(-1, *pixels.shape[2:])[valid].float().div_(255.0)
        values = (values - self.mean) / self.std
        with torch.no_grad():
            values = self.backbone.conv1(values)
            values = self.backbone.bn1(values)
            values = self.backbone.relu(values)
            values = self.backbone.maxpool(values)
            values = self.backbone.layer1(values)
            values = self.backbone.layer2(values)
            values = self.backbone.layer3(values)
        values = self.backbone.layer4(values)
        output[valid] = torch.flatten(self.backbone.avgpool(values), 1)
        return output.reshape(batch_size, steps, self.embedding_dim)


class TrajectoryPatchModel(nn.Module):
    input_names = ("trajectory", "patch_pixels", "patch_mask", "patch_quality")

    def __init__(
        self,
        trajectory_input_dim: int,
        hidden_dim: int = 128,
        trajectory_hidden_dim: int = 128,
        dropout: float = 0.25,
        num_event_types: int = 2,
        trajectory_conv_layers: int = 3,
        trajectory_gru_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % 2 or num_event_types <= 0 or not 0 <= dropout < 1:
            raise ValueError("模型维度、事件数或 dropout 无效")
        self.trajectory_encoder = _TrajectoryEncoder(
            trajectory_input_dim,
            trajectory_hidden_dim,
            dropout,
            trajectory_conv_layers,
            trajectory_gru_layers,
        )
        self.patch_backbone = _ResNet18Layer4Encoder()
        self.image_encoder = _PatchBranchEncoder(self.patch_backbone.embedding_dim, hidden_dim, dropout)
        fusion_dim = self.trajectory_encoder.embedding_dim + self.image_encoder.branch_embedding_dim + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.eventness_head = _head(hidden_dim, 1, dropout)
        self.type_head = _head(hidden_dim, num_event_types, dropout)

    def forward(
        self,
        *,
        trajectory: torch.Tensor,
        patch_pixels: torch.Tensor,
        patch_mask: torch.Tensor,
        patch_quality: torch.Tensor,
        patch_quality_continuous: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trajectory_embedding = self.trajectory_encoder(trajectory)
        patch_image = self.patch_backbone(patch_pixels, patch_mask)
        embeddings, ratios = self.image_encoder(
            patch_image=patch_image,
            patch_mask=patch_mask,
            patch_quality=patch_quality,
            patch_quality_continuous=patch_quality_continuous,
        )
        fused = self.fusion(torch.cat((trajectory_embedding, *embeddings, *ratios), dim=-1))
        return {
            "eventness_logit": self.eventness_head(fused).squeeze(-1),
            "type_logits": self.type_head(fused),
        }


def _head(hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _validate_mask(mask: torch.Tensor, values: torch.Tensor) -> None:
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape != values.shape[:2]:
        raise ValueError("patch_mask 必须为 [B,K] bool 且与输入对齐")
