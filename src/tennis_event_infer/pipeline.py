from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import LoadedCheckpoint, load_checkpoint
from .trajectory import load_v5_csv, make_trajectory_windows
from .types import Event, TrajectoryFrame
from .video import PatchReader, VideoInfo, scan_video


_EVENT_FIELDS = ("frame_number", "timestamp_seconds", "event_type", "confidence", "x", "y")
_MODEL_INPUTS = (
    "trajectory",
    "patch_pixels",
    "patch_mask",
    "patch_quality",
    "patch_quality_continuous",
)


def _prefetch_one(iterable):
    iterator = iter(iterable)
    finished = object()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(next, iterator, finished)
    try:
        while (item := future.result()) is not finished:
            future = executor.submit(next, iterator, finished)
            yield item
    finally:
        future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class FrameScore:
    frame_number: int
    hit: float
    bounce: float


class FrameDataset(Dataset):
    def __init__(
        self,
        video: str | Path,
        frames: list[TrajectoryFrame],
        video_info: VideoInfo,
        checkpoint: LoadedCheckpoint,
    ) -> None:
        if video_info.frame_count != len(frames):
            raise ValueError("视频帧数与 V5 轨迹帧数不一致")
        windows = make_trajectory_windows(
            frames,
            radius=checkpoint.trajectory.window_radius,
            fps=video_info.fps,
            max_gap_seconds=checkpoint.trajectory.max_derivative_gap_seconds,
        )
        self.trajectory = checkpoint.trajectory.normalizer.transform(windows)
        self.patch_reader = PatchReader(Path(video), frames, video_info.timeline, checkpoint.patch)

    def __len__(self) -> int:
        return len(self.trajectory)

    def __getitem__(self, frame_number: int) -> dict[str, torch.Tensor | int]:
        return {
            "frame_number": frame_number,
            "trajectory": torch.from_numpy(self.trajectory[frame_number]),
            **self.patch_reader.sequence(frame_number),
        }

    def close(self) -> None:
        self.patch_reader.close()


def run_inference(
    *,
    video: str | Path,
    trajectory: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    device: str,
    batch_size: int,
) -> dict[str, object]:
    pipeline_started = time.perf_counter()
    loaded = load_checkpoint(checkpoint, device)
    frames = load_v5_csv(trajectory)
    video_info = scan_video(video)
    input_preparation_finished = time.perf_counter()
    reader_stats: dict[str, int] = {}
    scores = predict_frame_scores(
        video, frames, video_info, loaded, batch_size=batch_size, reader_stats=reader_stats,
    )
    scoring_finished = time.perf_counter()
    events = scores_to_events(
        scores,
        thresholds=loaded.postprocess.thresholds,
        nms_radius=loaded.postprocess.nms_radius,
        timeline=video_info.timeline,
        frames=frames,
    )
    postprocess_finished = time.perf_counter()
    json_path, csv_path = write_events(events, output_dir)
    output_write_finished = time.perf_counter()
    return {
        "device": str(loaded.device),
        "frames": len(frames),
        "events": len(events),
        "events_json": json_path.resolve(),
        "events_csv": csv_path.resolve(),
        "inference_performance": {
            "input_preparation_seconds": input_preparation_finished - pipeline_started,
            "scoring_seconds": scoring_finished - input_preparation_finished,
            "postprocess_seconds": postprocess_finished - scoring_finished,
            "output_write_seconds": output_write_finished - postprocess_finished,
            "pipeline_total_seconds": output_write_finished - pipeline_started,
            **reader_stats,
        },
    }


def event_scores(output: dict[str, torch.Tensor]) -> torch.Tensor:
    if set(output) != {"eventness_logit", "type_logits"}:
        raise ValueError("模型输出必须且只能包含 eventness_logit/type_logits")
    eventness = torch.sigmoid(output["eventness_logit"])
    type_probabilities = torch.softmax(output["type_logits"], dim=-1)
    if eventness.ndim != 1 or type_probabilities.shape != (len(eventness), 2):
        raise ValueError("模型输出 shape 无效")
    return eventness.unsqueeze(-1) * type_probabilities


def predict_frame_scores(
    video: str | Path,
    frames: list[TrajectoryFrame],
    video_info: VideoInfo,
    checkpoint: LoadedCheckpoint,
    *,
    batch_size: int,
    reader_stats: dict[str, int] | None = None,
) -> list[FrameScore]:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    dataset = FrameDataset(video, frames, video_info, checkpoint)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=checkpoint.device.type == "cuda",
    )
    scores: list[FrameScore] = []
    try:
        checkpoint.model.eval()
        with torch.inference_mode():
            batches = _prefetch_one(loader) if checkpoint.device.type == "cuda" else loader
            for batch in batches:
                inputs = {
                    name: batch[name].to(checkpoint.device, non_blocking=checkpoint.device.type == "cuda")
                    for name in _MODEL_INPUTS
                }
                probabilities = event_scores(checkpoint.model(**inputs)).cpu()
                for frame_number, probability in zip(batch["frame_number"].tolist(), probabilities.tolist()):
                    scores.append(FrameScore(int(frame_number), float(probability[0]), float(probability[1])))
    finally:
        try:
            dataset.close()
        finally:
            if reader_stats is not None:
                reader_stats.update(dataset.patch_reader.stats)
    if len(scores) != len(frames):
        raise RuntimeError("模型分数没有逐帧覆盖视频")
    return scores


def scores_to_events(
    scores: Sequence[FrameScore],
    *,
    thresholds: Mapping[str, float],
    nms_radius: int,
    timeline: Sequence[float] | np.ndarray,
    frames: Sequence[TrajectoryFrame],
) -> list[Event]:
    if set(thresholds) != {"hit", "bounce"} or not all(_unit_finite(value) for value in thresholds.values()):
        raise ValueError("thresholds 必须包含 [0,1] 内的 hit/bounce")
    if not isinstance(nms_radius, int) or isinstance(nms_radius, bool) or nms_radius <= 0:
        raise ValueError("nms_radius 必须是正整数")
    times = np.asarray(timeline, dtype=np.float64)
    if times.shape != (len(frames),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timeline 必须逐帧覆盖且严格递增")
    frame_numbers = [score.frame_number for score in scores]
    if len(frame_numbers) != len(set(frame_numbers)) or any(not 0 <= number < len(frames) for number in frame_numbers):
        raise ValueError("scores 帧号必须唯一且位于视频范围内")

    events = []
    for event_type in ("hit", "bounce"):
        candidates = [score for score in scores if getattr(score, event_type) >= thresholds[event_type]]
        kept: list[FrameScore] = []
        for candidate in sorted(candidates, key=lambda item: (-getattr(item, event_type), item.frame_number)):
            if any(abs(candidate.frame_number - other.frame_number) <= nms_radius for other in kept):
                continue
            kept.append(candidate)
        for score in kept:
            frame = frames[score.frame_number]
            events.append(
                Event(
                    score.frame_number,
                    float(times[score.frame_number]),
                    event_type,
                    float(getattr(score, event_type)),
                    float(frame.x) if frame.detected else None,
                    float(frame.y) if frame.detected else None,
                )
            )
    return sorted(events, key=lambda event: (event.frame_number, event.event_type))


def write_events(events: Sequence[Event], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"输出路径不是目录: {output}")
        unexpected = {item.name for item in output.iterdir()} - {"events.json", "events.csv"}
        if unexpected:
            raise ValueError(f"输出目录包含无关文件，拒绝覆盖: {sorted(unexpected)}")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
    try:
        rows = [asdict(event) for event in events]
        _write_csv(staging / "events.csv", rows)
        _write_json(staging / "events.json", rows)
        _publish_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "events.json", output / "events.csv"


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EVENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(staging: Path, output: Path) -> None:
    if not output.exists():
        os.replace(staging, output)
        return
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def _unit_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
