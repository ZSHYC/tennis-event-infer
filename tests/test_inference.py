from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
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


def test_prefetch_one_preserves_order_and_loads_on_background_thread():
    caller = threading.get_ident()
    producers = []

    def items():
        for value in range(3):
            producers.append(threading.get_ident())
            yield value

    assert list(pipeline._prefetch_one(items())) == [0, 1, 2]
    assert producers and set(producers) != {caller}


def test_prefetch_one_propagates_producer_failure():
    def items():
        yield 1
        raise RuntimeError("producer failed")

    iterator = pipeline._prefetch_one(items())
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="producer failed"):
        next(iterator)


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
    assert isinstance(scores, list)
    assert scores == [FrameScore(index, 0.25, 0.25) for index in range(8)]


def test_predict_frame_scores_copies_reader_stats_when_close_fails(monkeypatch):
    class Reader:
        @property
        def stats(self):
            assert dataset.closed
            return {"decoded_frame_count": 7}

    class Dataset:
        patch_reader = Reader()
        closed = False

        def close(self):
            self.closed = True
            raise RuntimeError("release failed")

    class Model(torch.nn.Module):
        def forward(self, **inputs):
            return {"eventness_logit": torch.zeros(1), "type_logits": torch.zeros(1, 2)}

    dataset = Dataset()
    batch = {
        "frame_number": torch.tensor([0]),
        "trajectory": torch.zeros(1, 1, 11),
        "patch_pixels": torch.zeros(1, 5, 3, 8, 8, dtype=torch.uint8),
        "patch_mask": torch.ones(1, 5, dtype=torch.bool),
        "patch_quality": torch.ones(1, 5, dtype=torch.long),
        "patch_quality_continuous": torch.zeros(1, 5, 3),
    }
    loaded = SimpleNamespace(model=Model(), device=torch.device("cpu"))
    stats = {}
    monkeypatch.setattr(pipeline, "FrameDataset", lambda *args: dataset)
    monkeypatch.setattr(pipeline, "DataLoader", lambda *args, **kwargs: [batch])

    with pytest.raises(RuntimeError, match="release failed"):
        predict_frame_scores(None, _frames(1), None, loaded, batch_size=128, reader_stats=stats)

    assert stats == {"decoded_frame_count": 7}


def test_predict_frame_scores_closes_and_copies_stats_when_model_fails(monkeypatch):
    class Reader:
        @property
        def stats(self):
            assert dataset.closed
            return {"decoded_frame_count": 1}

    class Dataset:
        patch_reader = Reader()
        closed = False

        def close(self):
            self.closed = True

    class Model(torch.nn.Module):
        def forward(self, **inputs):
            raise RuntimeError("model failed")

    dataset = Dataset()
    batch = {
        "frame_number": torch.tensor([0]),
        **{name: torch.zeros(1) for name in pipeline._MODEL_INPUTS},
    }
    loaded = SimpleNamespace(model=Model(), device=torch.device("cpu"))
    stats = {}
    monkeypatch.setattr(pipeline, "FrameDataset", lambda *args: dataset)
    monkeypatch.setattr(pipeline, "DataLoader", lambda *args, **kwargs: [batch])

    with pytest.raises(RuntimeError, match="model failed"):
        predict_frame_scores(None, _frames(1), None, loaded, batch_size=128, reader_stats=stats)

    assert dataset.closed is True
    assert stats == {"decoded_frame_count": 1}


def test_run_inference_reports_phase_timings_and_reader_stats_without_new_files(tmp_path, monkeypatch):
    frames = _frames(1)
    video_info = SimpleNamespace(timeline=np.array([0.0]))
    loaded = SimpleNamespace(
        device=torch.device("cpu"),
        postprocess=SimpleNamespace(thresholds={"hit": 0.5, "bounce": 0.5}, nms_radius=1),
    )
    reader_stats = {
        "video_open_count": 1,
        "decoded_frame_count": 1,
        "backward_request_count": 0,
        "patch_cache_hits": 2,
        "patch_cache_misses": 1,
        "peak_cached_patches": 1,
    }
    monkeypatch.setattr(pipeline, "load_checkpoint", lambda *args: loaded)
    monkeypatch.setattr(pipeline, "load_v5_csv", lambda *args: frames)
    monkeypatch.setattr(pipeline, "scan_video", lambda *args: video_info)

    def predict(*args, batch_size, reader_stats=None):
        assert batch_size == 128
        assert reader_stats is not None
        reader_stats.update(reader_stats_fixture)
        return [FrameScore(0, 0.25, 0.25)]

    reader_stats_fixture = reader_stats
    monkeypatch.setattr(pipeline, "predict_frame_scores", predict)
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    clock = iter((10.0, 11.0, 13.0, 16.0, 20.0))
    output = tmp_path / "events"

    summary = pipeline.run_inference(
        video="video", trajectory="track.csv", checkpoint="model.pt", output_dir=output,
        device="cpu", batch_size=128,
    )

    assert set(summary) == {"device", "frames", "events", "events_json", "events_csv", "inference_performance"}
    assert {path.name for path in output.iterdir()} == {"events.json", "events.csv"}
    assert summary["inference_performance"] == {
        "input_preparation_seconds": 1.0,
        "scoring_seconds": 2.0,
        "postprocess_seconds": 3.0,
        "output_write_seconds": 4.0,
        "pipeline_total_seconds": 10.0,
        **reader_stats,
    }


@pytest.mark.parametrize(("batch_args", "expected"), [((), 128), (("--batch-size", "17"), 17)])
def test_cli_forwards_batch_size_and_prints_parseable_performance_json(
    tmp_path, monkeypatch, capsys, batch_args, expected,
):
    import tennis_event_infer.cli as cli

    received = {}
    performance = {"pipeline_total_seconds": 1.25, "decoded_frame_count": 3}

    def run(**kwargs):
        received.update(kwargs)
        return {
            "device": "cpu",
            "frames": 1,
            "events": 0,
            "events_json": tmp_path / "events.json",
            "events_csv": tmp_path / "events.csv",
            "inference_performance": performance,
        }

    monkeypatch.setattr(cli, "run_inference", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tennis-event-infer", "--video", "video", "--trajectory", "track.csv", "--checkpoint", "model.pt",
         "--output-dir", str(tmp_path), *batch_args],
    )

    cli.main()

    assert received["batch_size"] == expected
    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("inference_performance: ")
    ]
    assert len(lines) == 1
    line = lines[0]
    assert json.loads(line.removeprefix("inference_performance: ")) == performance


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
