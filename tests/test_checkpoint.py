from __future__ import annotations

import copy

import pytest
import torch

from tennis_event_infer.checkpoint import load_checkpoint
from tennis_event_infer.model import TrajectoryPatchModel
from tennis_event_infer.trajectory import FEATURE_NAMES


def _model():
    return TrajectoryPatchModel(
        trajectory_input_dim=11,
        hidden_dim=64,
        trajectory_hidden_dim=64,
        dropout=0.25,
        trajectory_conv_layers=3,
        trajectory_gru_layers=1,
        num_event_types=2,
    )


def test_model_has_only_trajectory_and_patch_inputs():
    model = _model().eval()
    assert model.input_names == ("trajectory", "patch_pixels", "patch_mask", "patch_quality")
    with torch.inference_mode():
        output = model(
            trajectory=torch.zeros(2, 25, 11),
            patch_pixels=torch.zeros(2, 5, 3, 224, 224, dtype=torch.uint8),
            patch_mask=torch.ones(2, 5, dtype=torch.bool),
            patch_quality=torch.full((2, 5), 3, dtype=torch.long),
            patch_quality_continuous=torch.zeros(2, 5, 3),
        )
    assert output["eventness_logit"].shape == (2,)
    assert output["type_logits"].shape == (2, 2)


def test_model_matches_frozen_epoch10_parameter_contract():
    model = _model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 11_919_099
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 8_618_195
    assert sum(tensor.numel() for tensor in model.state_dict().values()) == 11_928_725
    assert {name.split(".", 1)[0] for name in model.state_dict()} == {
        "trajectory_encoder",
        "patch_backbone",
        "image_encoder",
        "fusion",
        "eventness_head",
        "type_head",
    }


def _payload():
    return {
        "format_version": 1,
        "model_id": "trajectory_patch_resnet18_layer4_v1",
        "model_state": _model().state_dict(),
        "model_config": {
            "hidden_dim": 64,
            "trajectory_hidden_dim": 64,
            "dropout": 0.25,
            "trajectory_conv_layers": 3,
            "trajectory_gru_layers": 1,
            "num_event_types": 2,
        },
        "trajectory_contract": {
            "feature_schema_version": 3,
            "feature_names": list(FEATURE_NAMES),
            "window_radius": 12,
            "max_derivative_gap_seconds": 0.2,
            "normalizer": {
                "type": "robust",
                "feature_names": ["x_norm"],
                "medians": [0.5],
                "scales": [0.2],
                "clip": 5.0,
                "mask_feature": "detected",
            },
        },
        "patch_contract": {
            "size": [224, 224],
            "fill": [114, 114, 114],
            "radius_ratio": 0.08,
            "max_gap_seconds": 0.1,
            "offsets_seconds": [-0.2, -0.1, 0.0, 0.1, 0.2],
        },
        "postprocess": {
            "event_types": ["hit", "bounce"],
            "score_mode": "product",
            "thresholds": {"hit": 0.1, "bounce": 0.1},
            "nms_radius": 5,
        },
    }


def _save(tmp_path, payload=None):
    path = tmp_path / "checkpoint.pt"
    torch.save(payload or _payload(), path)
    return path


def test_loader_rejects_training_checkpoint(tmp_path):
    path = _save(tmp_path, {"model_type": "multimodal_resnet18_layer4_finetune"})
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path, device="cpu")


def test_loader_returns_model_and_immutable_contract(tmp_path):
    loaded = load_checkpoint(_save(tmp_path), device="cpu")
    assert isinstance(loaded.model, TrajectoryPatchModel)
    assert loaded.model.training is False
    assert loaded.postprocess.thresholds == {"hit": 0.1, "bounce": 0.1}
    assert loaded.patch.offsets_seconds == (-0.2, -0.1, 0.0, 0.1, 0.2)
    assert loaded.trajectory.feature_names == FEATURE_NAMES
    with pytest.raises(TypeError):
        loaded.postprocess.thresholds["hit"] = 0.2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("format_version",), 2),
        (("model_id",), "other"),
        (("trajectory_contract", "feature_schema_version"), 2),
        (("trajectory_contract", "window_radius"), 0),
        (("patch_contract", "offsets_seconds"), [-0.1, 0.0, 0.1]),
        (("postprocess", "event_types"), ["hit"]),
        (("postprocess", "score_mode"), "eventness"),
        (("postprocess", "thresholds", "hit"), float("nan")),
    ],
)
def test_loader_rejects_invalid_contract(tmp_path, path, value):
    payload = copy.deepcopy(_payload())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        load_checkpoint(_save(tmp_path, payload), device="cpu")


def test_loader_rejects_training_field_and_wrong_state_dict(tmp_path):
    payload = _payload()
    payload["optimizer_state"] = {}
    with pytest.raises(ValueError, match="顶层字段"):
        load_checkpoint(_save(tmp_path, payload), device="cpu")

    payload = _payload()
    payload["model_state"].pop("fusion.0.weight")
    with pytest.raises(ValueError, match="state dict"):
        load_checkpoint(_save(tmp_path, payload), device="cpu")

    payload = _payload()
    payload["model_state"] = []
    with pytest.raises(ValueError, match="state dict"):
        load_checkpoint(_save(tmp_path, payload), device="cpu")


def test_loader_rejects_unknown_device(tmp_path):
    with pytest.raises(ValueError, match="device"):
        load_checkpoint(_save(tmp_path), device="tpu")
