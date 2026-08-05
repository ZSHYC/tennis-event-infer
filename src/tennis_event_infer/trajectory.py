from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .types import TrajectoryFrame


V5_FIELDS = {"frame_number", "detected", "x_orig", "y_orig", "width", "height"}
FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "detected",
    "dx",
    "dy",
    "speed",
    "acceleration",
    "angle_change",
    "curvature",
    "valid",
    "time_offset",
)
FEATURE_SCHEMA_VERSION = 3
_EPSILON = 1e-6


def load_v5_csv(path: str | Path) -> list[TrajectoryFrame]:
    source = Path(path)
    frames: list[TrajectoryFrame] = []
    resolution: tuple[int, int] | None = None
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(V5_FIELDS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"V5 CSV 缺少字段 {missing}: {source}")
        for row_number, row in enumerate(reader, start=2):
            try:
                frame_number = int(row["frame_number"])
                detected_raw = int(row["detected"])
                width = int(row["width"])
                height = int(row["height"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"V5 CSV 整数字段无效: {source}:{row_number}") from exc
            if detected_raw not in (0, 1):
                raise ValueError(f"detected 必须是 0 或 1: {source}:{row_number}")
            if width <= 0 or height <= 0:
                raise ValueError(f"分辨率必须是正整数: {source}:{row_number}")
            if resolution is None:
                resolution = (width, height)
            elif resolution != (width, height):
                raise ValueError(f"V5 CSV 分辨率跨帧不一致: {source}:{row_number}")

            detected = bool(detected_raw)
            x = y = None
            if detected:
                try:
                    x = float(row["x_orig"])
                    y = float(row["y_orig"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"检测帧坐标无效: {source}:{row_number}") from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"检测帧坐标必须有限: {source}:{row_number}")
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(f"检测帧坐标超出画面范围: {source}:{row_number}")
            frames.append(TrajectoryFrame(frame_number, detected, x, y, width, height))

    if not frames:
        raise ValueError(f"V5 CSV 不得为空: {source}")
    if [frame.frame_number for frame in frames] != list(range(len(frames))):
        raise ValueError(f"V5 CSV 必须逐帧覆盖 0..N-1: {source}")
    return frames


def make_trajectory_windows(
    frames: list[TrajectoryFrame],
    *,
    radius: int,
    fps: float,
    max_gap_seconds: float,
) -> np.ndarray:
    if not isinstance(radius, int) or isinstance(radius, bool) or radius <= 0:
        raise ValueError("radius 必须是正整数")
    if not _positive_finite(fps):
        raise ValueError("fps 必须是有限正数")
    if not _positive_finite(max_gap_seconds):
        raise ValueError("max_gap_seconds 必须是有限正数")
    if [frame.frame_number for frame in frames] != list(range(len(frames))):
        raise ValueError("轨迹帧必须逐帧覆盖 0..N-1")

    rows = _feature_rows(frames, fps, max_gap_seconds)
    windows = np.zeros((len(frames), radius * 2 + 1, len(FEATURE_NAMES)), dtype=np.float32)
    time_index = FEATURE_NAMES.index("time_offset")
    for center in range(len(frames)):
        for offset in range(-radius, radius + 1):
            position = offset + radius
            frame_number = center + offset
            if 0 <= frame_number < len(frames):
                windows[center, position] = rows[frame_number]
            windows[center, position, time_index] = offset / radius
    return windows


def _feature_rows(frames: list[TrajectoryFrame], fps: float, max_gap_seconds: float) -> np.ndarray:
    rows = np.zeros((len(frames), len(FEATURE_NAMES)), dtype=np.float32)
    previous_x = previous_y = previous_speed = previous_angle = None
    previous_frame = None
    for index, frame in enumerate(frames):
        detected = frame.detected and frame.x is not None and frame.y is not None
        x_norm = frame.x / frame.width if detected else 0.0
        y_norm = frame.y / frame.height if detected else 0.0
        dx = dy = speed = acceleration = angle_change = curvature = 0.0

        if detected and previous_x is not None and previous_y is not None and previous_frame is not None:
            elapsed = (frame.frame_number - previous_frame) / fps
            if elapsed <= max_gap_seconds:
                dx = (x_norm - previous_x) / elapsed
                dy = (y_norm - previous_y) / elapsed
                speed = math.hypot(dx, dy)
                acceleration = (speed - (previous_speed or 0.0)) / elapsed
                angle = math.atan2(dy, dx) if speed else previous_angle
                delta = _angle_delta(angle, previous_angle) if angle is not None and previous_angle is not None else 0.0
                angle_change = delta / math.pi
                curvature = math.log1p((delta / elapsed) / max(speed, _EPSILON))
                previous_speed, previous_angle = speed, angle
            else:
                previous_speed, previous_angle = 0.0, None
        elif detected:
            previous_speed, previous_angle = 0.0, None

        if detected:
            previous_x, previous_y = x_norm, y_norm
            previous_frame = frame.frame_number
        rows[index] = (
            x_norm,
            y_norm,
            float(detected),
            dx,
            dy,
            speed,
            acceleration,
            angle_change,
            curvature,
            1.0,
            0.0,
        )
    return rows


@dataclass(frozen=True)
class FeatureNormalizer:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    clip: float
    mask_feature: str | None

    @classmethod
    def from_contract(cls, contract: dict) -> "FeatureNormalizer":
        expected = {"type", "feature_names", "medians", "scales", "clip", "mask_feature"}
        if not isinstance(contract, dict) or set(contract) != expected or contract.get("type") != "robust":
            raise ValueError("normalizer 合同无效")
        names = tuple(contract["feature_names"])
        medians = tuple(float(value) for value in contract["medians"])
        scales = tuple(float(value) for value in contract["scales"])
        if not names or len(names) != len(medians) or len(names) != len(scales):
            raise ValueError("normalizer 统计量长度不一致")
        if len(set(names)) != len(names) or any(name not in FEATURE_NAMES for name in names):
            raise ValueError("normalizer feature_names 无效")
        if not all(math.isfinite(value) for value in medians):
            raise ValueError("normalizer medians 必须有限")
        if not all(_positive_finite(value) for value in scales):
            raise ValueError("normalizer scales 必须是有限正数")
        clip = float(contract["clip"])
        if not _positive_finite(clip):
            raise ValueError("normalizer clip 必须是有限正数")
        mask_feature = contract["mask_feature"]
        if mask_feature is not None and mask_feature not in FEATURE_NAMES:
            raise ValueError("normalizer mask_feature 无效")
        return cls(names, medians, scales, clip, mask_feature)

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=np.float32).copy()
        if transformed.ndim < 1 or transformed.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(f"轨迹特征形状无效: {transformed.shape}")
        for name, median, scale in zip(self.feature_names, self.medians, self.scales):
            index = FEATURE_NAMES.index(name)
            transformed[..., index] = np.clip((transformed[..., index] - median) / scale, -self.clip, self.clip)
        if self.mask_feature is not None:
            mask = transformed[..., FEATURE_NAMES.index(self.mask_feature)] <= 0.5
            for name in self.feature_names:
                transformed[..., FEATURE_NAMES.index(name)][mask] = 0.0
        return transformed


def _angle_delta(angle: float, previous: float) -> float:
    delta = abs(angle - previous)
    return min(delta, 2 * math.pi - delta)


def _positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
