from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

from .model import TrajectoryPatchModel
from .trajectory import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureNormalizer


_TOP_LEVEL = {
    "format_version",
    "model_id",
    "model_state",
    "model_config",
    "trajectory_contract",
    "patch_contract",
    "postprocess",
}
_MODEL_CONFIG = {
    "hidden_dim",
    "trajectory_hidden_dim",
    "dropout",
    "trajectory_conv_layers",
    "trajectory_gru_layers",
    "num_event_types",
}


@dataclass(frozen=True)
class TrajectoryContract:
    feature_names: tuple[str, ...]
    window_radius: int
    max_derivative_gap_seconds: float
    normalizer: FeatureNormalizer


@dataclass(frozen=True)
class PatchContract:
    size: tuple[int, int]
    fill: tuple[int, int, int]
    radius_ratio: float
    max_gap_seconds: float
    offsets_seconds: tuple[float, ...]


@dataclass(frozen=True)
class PostprocessContract:
    event_types: tuple[str, str]
    score_mode: str
    thresholds: Mapping[str, float]
    nms_radius: int


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: TrajectoryPatchModel
    trajectory: TrajectoryContract
    patch: PatchContract
    postprocess: PostprocessContract
    device: torch.device


def load_checkpoint(path: str | Path, device: str = "auto") -> LoadedCheckpoint:
    resolved_device = _device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("deployment checkpoint 顶层必须是字典并包含 format_version")
    if set(payload) != _TOP_LEVEL:
        missing = sorted(_TOP_LEVEL - set(payload))
        extra = sorted(set(payload) - _TOP_LEVEL)
        raise ValueError(f"deployment checkpoint 顶层字段无效；缺少={missing}，多余={extra}")
    if payload["format_version"] != 1:
        raise ValueError(f"不支持 format_version={payload['format_version']!r}")
    if payload["model_id"] != "trajectory_patch_resnet18_layer4_v1":
        raise ValueError(f"不支持 model_id={payload['model_id']!r}")

    trajectory = _trajectory_contract(payload["trajectory_contract"])
    patch = _patch_contract(payload["patch_contract"])
    postprocess = _postprocess_contract(payload["postprocess"])
    config = _model_config(payload["model_config"])
    if config["num_event_types"] != len(postprocess.event_types):
        raise ValueError("model_config num_event_types 与 event_types 不一致")

    model = TrajectoryPatchModel(trajectory_input_dim=len(trajectory.feature_names), **config)
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"model state dict 与 {payload['model_id']} 不兼容") from exc
    model.to(resolved_device)
    model.eval()
    return LoadedCheckpoint(model, trajectory, patch, postprocess, resolved_device)


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda，但当前环境没有可用 CUDA")
    return torch.device(value)


def _trajectory_contract(value) -> TrajectoryContract:
    fields = {
        "feature_schema_version",
        "feature_names",
        "window_radius",
        "max_derivative_gap_seconds",
        "normalizer",
    }
    contract = _exact_dict(value, fields, "trajectory_contract")
    if contract["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature_schema_version 不受支持")
    names = tuple(contract["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("feature_names 与模型 schema 不一致")
    radius = contract["window_radius"]
    if not _positive_int(radius):
        raise ValueError("window_radius 必须是正整数")
    max_gap = contract["max_derivative_gap_seconds"]
    if not _positive_finite(max_gap):
        raise ValueError("max_derivative_gap_seconds 必须是有限正数")
    normalizer = FeatureNormalizer.from_contract(contract["normalizer"])
    return TrajectoryContract(names, radius, float(max_gap), normalizer)


def _patch_contract(value) -> PatchContract:
    fields = {"size", "fill", "radius_ratio", "max_gap_seconds", "offsets_seconds"}
    contract = _exact_dict(value, fields, "patch_contract")
    size = tuple(contract["size"])
    if len(size) != 2 or not all(_positive_int(item) for item in size):
        raise ValueError("patch size 必须包含两个正整数")
    fill = tuple(contract["fill"])
    if len(fill) != 3 or not all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= 255 for item in fill):
        raise ValueError("patch fill 必须包含三个 0..255 整数")
    radius_ratio = contract["radius_ratio"]
    max_gap = contract["max_gap_seconds"]
    if not _positive_finite(radius_ratio) or radius_ratio > 1:
        raise ValueError("radius_ratio 必须在 (0,1] 内")
    if not _positive_finite(max_gap):
        raise ValueError("max_gap_seconds 必须是有限正数")
    offsets = tuple(float(item) for item in contract["offsets_seconds"])
    if len(offsets) != 5 or not all(math.isfinite(item) for item in offsets):
        raise ValueError("offsets_seconds 必须包含五个有限数值")
    return PatchContract(size, fill, float(radius_ratio), float(max_gap), offsets)


def _postprocess_contract(value) -> PostprocessContract:
    fields = {"event_types", "score_mode", "thresholds", "nms_radius"}
    contract = _exact_dict(value, fields, "postprocess")
    event_types = tuple(contract["event_types"])
    if event_types != ("hit", "bounce"):
        raise ValueError("event_types 必须是 hit/bounce")
    if contract["score_mode"] != "product":
        raise ValueError("score_mode 必须是 product")
    thresholds = contract["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != set(event_types):
        raise ValueError("thresholds 必须包含且仅包含 hit/bounce")
    if not all(_unit_finite(item) for item in thresholds.values()):
        raise ValueError("thresholds 必须是 [0,1] 内有限数值")
    nms_radius = contract["nms_radius"]
    if not _positive_int(nms_radius):
        raise ValueError("nms_radius 必须是正整数")
    immutable_thresholds = MappingProxyType({name: float(thresholds[name]) for name in event_types})
    return PostprocessContract(event_types, "product", immutable_thresholds, nms_radius)


def _model_config(value) -> dict:
    config = _exact_dict(value, _MODEL_CONFIG, "model_config")
    for name in ("hidden_dim", "trajectory_hidden_dim", "trajectory_conv_layers", "trajectory_gru_layers"):
        if not _positive_int(config[name]):
            raise ValueError(f"model_config {name} 必须是正整数")
    if not _positive_int(config["num_event_types"]):
        raise ValueError("model_config num_event_types 必须是正整数")
    if not _unit_finite(config["dropout"]) or config["dropout"] == 1:
        raise ValueError("model_config dropout 必须位于 [0,1)")
    return dict(config)


def _exact_dict(value, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} 字段无效")
    return value


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _unit_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
