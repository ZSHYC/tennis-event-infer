from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import tennis_event_infer.pipeline as pipeline
from tennis_event_infer.pipeline import FrameScore, event_scores, predict_frame_scores, scores_to_events, write_events
from tennis_event_infer.model import TrajectoryPatchModel
from tennis_event_infer.trajectory import FEATURE_NAMES, FeatureNormalizer
from tennis_event_infer.types import Event, TrajectoryFrame
from tennis_event_infer.video import scan_video


def _frames(count: int, missing: set[int] | None = None, size: tuple[int, int] = (100, 100)):
    missing = missing or set()
    return [
        TrajectoryFrame(
            index,
            index not in missing,
            None if index in missing else float(index),
            None if index in missing else float(index + 1),
            size[0],
            size[1],
        )
        for index in range(count)
    ]


def test_product_event_scores_are_sigmoid_times_softmax():
    output = {
        "eventness_logit": torch.tensor([0.0]),
        "type_logits": torch.tensor([[0.0, 0.0]]),
    }
    np.testing.assert_allclose(event_scores(output).numpy(), [[0.25, 0.25]])


def test_nms_is_per_class_and_prefers_higher_score():
    scores = [
        FrameScore(10, hit=0.7, bounce=0.1),
        FrameScore(12, hit=0.9, bounce=0.8),
        FrameScore(20, hit=0.8, bounce=0.2),
    ]
    events = scores_to_events(
        scores,
        thresholds={"hit": 0.1, "bounce": 0.1},
        nms_radius=5,
        timeline=np.arange(30, dtype=np.float64) / 10,
        frames=_frames(30),
    )
    assert [(event.frame_number, event.event_type) for event in events] == [
        (12, "bounce"),
        (12, "hit"),
        (20, "bounce"),
        (20, "hit"),
    ]


def test_nms_tie_uses_earlier_frame_and_missing_v5_coordinates_stay_null():
    events = scores_to_events(
        [FrameScore(3, hit=0.5, bounce=0.0), FrameScore(4, hit=0.5, bounce=0.0)],
        thresholds={"hit": 0.5, "bounce": 0.5},
        nms_radius=2,
        timeline=np.arange(8, dtype=np.float64) * 0.25,
        frames=_frames(8, missing={3}),
    )
    assert len(events) == 1
    assert events[0] == Event(3, 0.75, "hit", 0.5, None, None)


def test_write_events_publishes_only_fixed_json_and_csv(tmp_path):
    output = tmp_path / "result"
    events = [Event(3, 0.75, "hit", 0.5, None, None)]
    json_path, csv_path = write_events(events, output)
    assert {item.name for item in output.iterdir()} == {"events.json", "events.csv"}
    assert json.loads(json_path.read_text(encoding="utf-8")) == [
        {
            "frame_number": 3,
            "timestamp_seconds": 0.75,
            "event_type": "hit",
            "confidence": 0.5,
            "x": None,
            "y": None,
        }
    ]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))[0]["x"] == ""


def test_write_events_keeps_existing_pair_when_second_write_fails(tmp_path, monkeypatch):
    output = tmp_path / "result"
    write_events([Event(1, 0.1, "hit", 0.6, 4.0, 5.0)], output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def fail(*_args, **_kwargs):
        raise OSError("simulated")

    monkeypatch.setattr(pipeline, "_write_json", fail)
    with pytest.raises(OSError, match="simulated"):
        write_events([Event(2, 0.2, "bounce", 0.7, 6.0, 7.0)], output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert not list(tmp_path.glob(".result.*"))


def test_predict_frame_scores_runs_ordered_batches_without_workers(tmp_path):
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 8))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((8, 16, 3), index, dtype=np.uint8))
    writer.release()

    class DummyModel(torch.nn.Module):
        def forward(self, **inputs):
            batch = inputs["trajectory"].shape[0]
            return {"eventness_logit": torch.zeros(batch), "type_logits": torch.zeros(batch, 2)}

    normalizer = FeatureNormalizer.from_contract(
        {
            "type": "robust",
            "feature_names": ["x_norm"],
            "medians": [0.0],
            "scales": [1.0],
            "clip": 5.0,
            "mask_feature": "detected",
        }
    )
    loaded = SimpleNamespace(
        model=DummyModel(),
        device=torch.device("cpu"),
        trajectory=SimpleNamespace(window_radius=1, max_derivative_gap_seconds=0.2, normalizer=normalizer),
        patch=SimpleNamespace(
            size=(8, 8),
            fill=(114, 114, 114),
            radius_ratio=0.25,
            max_gap_seconds=0.2,
            offsets_seconds=(-0.2, -0.1, 0.0, 0.1, 0.2),
        ),
    )
    scores = predict_frame_scores(video, _frames(8, size=(16, 8)), scan_video(video), loaded, batch_size=3)
    assert scores == [FrameScore(index, 0.25, 0.25) for index in range(8)]


def _cli_command(*args: str):
    return [sys.executable, "-m", "tennis_event_infer.cli", *args]


def test_cli_help_exposes_only_operational_arguments():
    result = subprocess.run(_cli_command("--help"), check=False, capture_output=True, text=True)
    assert result.returncode == 0
    for name in ("--video", "--trajectory", "--checkpoint", "--output-dir", "--device", "--batch-size"):
        assert name in result.stdout
    for forbidden in ("--config", "--threshold", "--nms", "--cache", "--gt", "--visualize"):
        assert forbidden not in result.stdout


def test_cli_reports_input_and_unavailable_cuda_errors_without_traceback(tmp_path):
    missing = tmp_path / "missing"
    args = (
        "--video",
        str(missing),
        "--trajectory",
        str(missing),
        "--checkpoint",
        str(missing),
        "--output-dir",
        str(tmp_path / "out"),
    )
    result = subprocess.run(_cli_command(*args), check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Traceback" not in result.stderr

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(_cli_command(*args, "--device", "cuda"), env=env, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "没有可用 CUDA" in result.stderr
    assert "Traceback" not in result.stderr


def _write_deployment_checkpoint(path: Path):
    model = TrajectoryPatchModel(11, 64, 64, 0.25, 2, 3, 1)
    torch.save(
        {
            "format_version": 1,
            "model_id": "trajectory_patch_resnet18_layer4_v1",
            "model_state": model.state_dict(),
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
        },
        path,
    )


def test_cpu_cli_runs_short_video_end_to_end(tmp_path):
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 8))
    assert writer.isOpened()
    for index in range(16):
        writer.write(np.full((8, 16, 3), index, dtype=np.uint8))
    writer.release()
    trajectory = tmp_path / "track.csv"
    trajectory.write_text(
        "frame_number,detected,x_orig,y_orig,width,height,fps\n"
        + "".join(f"{index},1,{4 + index % 8},3,16,8,10\n" for index in range(16)),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "deploy.pt"
    _write_deployment_checkpoint(checkpoint)
    output = tmp_path / "output"
    result = subprocess.run(
        _cli_command(
            "--video",
            str(video),
            "--trajectory",
            str(trajectory),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "8",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == ["events.csv", "events.json"]
    assert "device: cpu" in result.stdout
    assert "events_json:" in result.stdout
