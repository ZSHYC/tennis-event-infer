from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np
import torch

from .types import TrajectoryFrame

if TYPE_CHECKING:
    from .checkpoint import PatchContract


QUALITY_IDS = {"missing": 0, "nearest": 1, "interpolated": 2, "observed": 3}


@dataclass(frozen=True)
class VideoInfo:
    timeline: np.ndarray
    timestamp_source: str
    fps: float
    width: int
    height: int
    frame_count: int


def scan_video(path: str | Path) -> VideoInfo:
    video = Path(path)
    if not video.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video}")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = _positive_dimension(capture.get(cv2.CAP_PROP_FRAME_WIDTH), "宽度")
        height = _positive_dimension(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), "高度")
        reported = []
        while capture.grab():
            reported.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        capture.release()
    if not reported:
        raise ValueError(f"视频没有可解码帧: {video}")
    if not _positive_finite(fps):
        raise ValueError(f"视频 FPS 必须是有限正数: {fps}")
    use_pts = all(math.isfinite(value) and value >= 0 for value in reported) and all(
        current > previous for previous, current in zip(reported, reported[1:])
    )
    timeline = np.asarray(reported if use_pts else np.arange(len(reported)) / fps, dtype=np.float64)
    return VideoInfo(timeline, "pts" if use_pts else "fps_fallback", fps, width, height, len(reported))


def nearest_frame_indices(
    timeline: Sequence[float] | np.ndarray,
    *,
    center: int,
    offsets: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    times = _timeline(timeline)
    if not isinstance(center, int) or isinstance(center, bool) or not 0 <= center < len(times):
        raise IndexError("center 超出视频时间轴")
    values = np.asarray(tuple(offsets), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("offsets 必须是一维有限数值")
    targets = times[center] + values
    valid = (targets >= times[0]) & (targets <= times[-1])
    indices = np.zeros(len(values), dtype=np.int64)
    if valid.any():
        target = targets[valid]
        right = np.minimum(np.searchsorted(times, target, side="left"), len(times) - 1)
        left = np.maximum(right - 1, 0)
        left_distance = target - times[left]
        right_distance = times[right] - target
        choose_left = left_distance <= right_distance + 1e-12
        indices[valid] = np.where(choose_left, left, right)
    return indices, valid.astype(np.bool_, copy=False)


def locate_patch_center_arrays(
    frames: Sequence[TrajectoryFrame],
    *,
    max_gap_seconds: float,
    timeline: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("max_gap_seconds 必须是有限非负数")
    ordered = list(frames)
    if [frame.frame_number for frame in ordered] != list(range(len(ordered))):
        raise ValueError("轨迹帧必须逐帧覆盖 0..N-1")
    times = _timeline(timeline, len(ordered))
    count = len(ordered)
    x = np.full(count, np.nan, dtype=np.float32)
    y = np.full(count, np.nan, dtype=np.float32)
    quality = np.zeros(count, dtype=np.uint8)
    distance = np.zeros(count, dtype=np.float32)
    observed = np.fromiter((frame.detected for frame in ordered), dtype=np.bool_, count=count)
    observed_indices = np.flatnonzero(observed)
    for index in observed_indices:
        x[index] = float(ordered[index].x)
        y[index] = float(ordered[index].y)
        quality[index] = QUALITY_IDS["observed"]
    if not len(observed_indices):
        return x, y, quality, distance

    left_position = 0
    for index in range(count):
        while left_position + 1 < len(observed_indices) and observed_indices[left_position + 1] <= index:
            left_position += 1
        left = int(observed_indices[left_position]) if observed_indices[left_position] <= index else None
        right_position = left_position if observed_indices[left_position] >= index else left_position + 1
        right = int(observed_indices[right_position]) if right_position < len(observed_indices) else None
        if observed[index]:
            continue
        left_gap = times[index] - times[left] if left is not None else math.inf
        right_gap = times[right] - times[index] if right is not None else math.inf
        left_valid = left_gap <= max_gap_seconds
        right_valid = right_gap <= max_gap_seconds
        if left_valid and right_valid:
            weight = left_gap / (times[right] - times[left])
            x[index] = float(ordered[left].x) + (float(ordered[right].x) - float(ordered[left].x)) * weight
            y[index] = float(ordered[left].y) + (float(ordered[right].y) - float(ordered[left].y)) * weight
            quality[index] = QUALITY_IDS["interpolated"]
            distance[index] = max(left_gap, right_gap)
        elif left_valid or right_valid:
            nearest = left if left_valid else right
            x[index], y[index] = ordered[nearest].x, ordered[nearest].y
            quality[index] = QUALITY_IDS["nearest"]
            distance[index] = min(left_gap, right_gap)
    return x, y, quality, distance


def crop_patch(image: np.ndarray, center: Sequence[float], radius: int, fill: Sequence[int]) -> np.ndarray:
    _validate_image(image)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius <= 0:
        raise ValueError("radius 必须是正整数")
    values = tuple(center)
    if len(values) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("center 必须包含两个有限数值")
    fill_values = _fill(fill, image.dtype)
    size = radius * 2
    output = np.empty((size, size, 3), dtype=image.dtype)
    output[...] = fill_values
    left, top, source_left, source_top, source_right, source_bottom = _crop_bounds(
        values, radius, image.shape[1], image.shape[0]
    )
    if source_left < source_right and source_top < source_bottom:
        destination_left = source_left - left
        destination_top = source_top - top
        output[
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = image[source_top:source_bottom, source_left:source_right]
    return output


def patch_valid_area_ratio(image_shape: Sequence[int], center: Sequence[float], radius: int) -> float:
    if len(image_shape) < 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in image_shape[:2]):
        raise ValueError("image_shape 必须以正整数 height/width 开头")
    if not isinstance(radius, int) or isinstance(radius, bool) or radius <= 0:
        raise ValueError("radius 必须是正整数")
    height, width = int(image_shape[0]), int(image_shape[1])
    _, _, left, top, right, bottom = _crop_bounds(center, radius, width, height)
    return max(0, right - left) * max(0, bottom - top) / float((2 * radius) ** 2)


def letterbox_image(image: np.ndarray, size: Sequence[int], fill: Sequence[int]) -> np.ndarray:
    _validate_image(image)
    width, height = _size(size)
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    output = np.empty((height, width, 3), dtype=image.dtype)
    output[...] = _fill(fill, image.dtype)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    return output


class PatchReader:
    def __init__(
        self,
        video: Path,
        frames: list[TrajectoryFrame],
        timeline: np.ndarray,
        contract: PatchContract,
        cache_size: int = 256,
    ) -> None:
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size <= 0:
            raise ValueError("cache_size 必须是正整数")
        self.video = Path(video)
        self.frames = list(frames)
        self.timeline = _timeline(timeline, len(self.frames)).copy()
        if [frame.frame_number for frame in self.frames] != list(range(len(self.frames))):
            raise ValueError("轨迹帧必须逐帧覆盖 0..N-1")
        self.size = _size(contract.size)
        self.fill = tuple(int(value) for value in contract.fill)
        _fill(self.fill, np.dtype(np.uint8))
        if not _positive_finite(contract.radius_ratio) or contract.radius_ratio > 1:
            raise ValueError("radius_ratio 必须在 (0,1] 内")
        self.offsets = tuple(float(value) for value in contract.offsets_seconds)
        if len(self.offsets) != 5 or not all(math.isfinite(value) for value in self.offsets):
            raise ValueError("offsets_seconds 必须包含五个有限值")

        capture = cv2.VideoCapture(str(self.video))
        try:
            if not capture.isOpened():
                raise ValueError(f"无法打开视频: {self.video}")
            width = _positive_dimension(capture.get(cv2.CAP_PROP_FRAME_WIDTH), "宽度")
            height = _positive_dimension(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), "高度")
            reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if reported_count > 0 and reported_count != len(self.frames):
            raise ValueError(f"视频帧数 {reported_count} 与轨迹帧数 {len(self.frames)} 不一致")
        if any((frame.width, frame.height) != (width, height) for frame in self.frames):
            raise ValueError("视频与轨迹分辨率不一致")
        if not _positive_finite(fps):
            raise ValueError("视频 FPS 必须是有限正数")

        self.width, self.height, self.fps = width, height, fps
        self.radius = max(1, int(round(contract.radius_ratio * min(width, height))))
        self.location_x, self.location_y, self.quality, self.location_distance = locate_patch_center_arrays(
            self.frames, max_gap_seconds=contract.max_gap_seconds, timeline=self.timeline
        )
        self.cache_size = cache_size
        self._indices, self._in_video = _nearest_frame_plan(self.timeline, self.offsets)
        planned_sources = self._indices[self._in_video]
        planned_sources = planned_sources[self.quality[planned_sources] != QUALITY_IDS["missing"]]
        references = np.bincount(planned_sources, minlength=len(self.frames)).astype(np.int64, copy=False)
        remaining = references.copy()
        live: set[int] = set()
        next_source = 0
        peak = 0
        for center in range(len(self.frames)):
            selected = self._indices[center][
                self._in_video[center]
                & (self.quality[self._indices[center]] != QUALITY_IDS["missing"])
            ]
            if len(selected):
                target = int(selected.max())
                for source in range(next_source, target + 1):
                    if remaining[source] > 0 and self.quality[source] != QUALITY_IDS["missing"]:
                        live.add(source)
                next_source = max(next_source, target + 1)
                peak = max(peak, len(live))
                for source in selected:
                    remaining[source] -= 1
                    if remaining[source] == 0:
                        live.remove(int(source))
        if peak > cache_size:
            raise ValueError(f"顺序 patch 计划峰值 {peak} 超过 cache_size={cache_size}，尚未开始视频解码")

        self._remaining_references: np.ndarray | None = references
        self._patches: dict[int, tuple[np.ndarray, float]] = {}
        self._capture: cv2.VideoCapture | None = None
        self._next_frame = 0
        self._next_center = 0
        self._closed = False
        self._stats = {
            "video_open_count": 0,
            "decoded_frame_count": 0,
            "backward_request_count": 0,
            "patch_cache_hits": 0,
            "patch_cache_misses": 0,
            "peak_cached_patches": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def sequence(self, center_frame: int) -> dict[str, torch.Tensor]:
        if self._closed:
            raise ValueError("PatchReader 已关闭")
        if not isinstance(center_frame, int) or isinstance(center_frame, bool) or center_frame != self._next_center:
            in_range = isinstance(center_frame, int) and not isinstance(center_frame, bool) and 0 <= center_frame < len(self.frames)
            source = int(self._indices[center_frame, 0]) if in_range else -1
            self._stats["backward_request_count"] += 1
            raise ValueError(f"中心帧必须严格递增: center={center_frame}, expected={self._next_center}, source={source}")
        indices = self._indices[center_frame]
        in_video = self._in_video[center_frame]
        selected = indices[in_video & (self.quality[indices] != QUALITY_IDS["missing"])]
        if len(selected):
            target = int(selected.max())
            missing = [int(source) for source in np.unique(selected) if int(source) not in self._patches]
            if any(source < self._next_frame for source in missing):
                source = next(source for source in missing if source < self._next_frame)
                self._stats["backward_request_count"] += 1
                raise RuntimeError(f"顺序解码遇到后退请求: center={center_frame}, source={source}")
            self._decode_through(target)

        width, height = self.size
        pixels = np.zeros((len(indices), 3, height, width), dtype=np.uint8)
        mask = np.zeros(len(indices), dtype=np.bool_)
        quality = np.zeros(len(indices), dtype=np.int64)
        continuous = np.zeros((len(indices), 3), dtype=np.float32)
        for output_index, frame_index in enumerate(indices):
            if not in_video[output_index]:
                continue
            quality_id = int(self.quality[frame_index])
            quality[output_index] = quality_id
            if quality_id == QUALITY_IDS["missing"]:
                continue
            patch, area = self._cached(int(frame_index))
            prepared = letterbox_image(patch, self.size, self.fill)
            pixels[output_index] = np.moveaxis(prepared[..., ::-1], -1, 0)
            mask[output_index] = True
            continuous[output_index] = (
                abs(float(self.timeline[frame_index]) - frame_index / self.fps),
                float(self.location_distance[frame_index]),
                area,
            )
        for frame_index in selected:
            self._remaining_references[frame_index] -= 1
            if self._remaining_references[frame_index] == 0:
                self._patches.pop(int(frame_index))
        self._next_center += 1
        return {
            "patch_pixels": torch.from_numpy(pixels),
            "patch_mask": torch.from_numpy(mask),
            "patch_quality": torch.from_numpy(quality),
            "patch_quality_continuous": torch.from_numpy(continuous),
        }

    def close(self) -> None:
        if self._closed:
            return
        capture, self._capture = self._capture, None
        self._closed = True
        self._patches.clear()
        self._indices = None
        self._in_video = None
        self._remaining_references = None
        if capture is not None:
            capture.release()

    def _cached(self, frame_number: int) -> tuple[np.ndarray, float]:
        if frame_number not in self._patches:
            self._stats["backward_request_count"] += 1
            raise RuntimeError(f"顺序 patch 缓存缺失: center={self._next_center}, source={frame_number}")
        self._stats["patch_cache_hits"] += 1
        return self._patches[frame_number]

    def _prepare(self, frame_number: int, image: np.ndarray) -> None:
        center = (float(self.location_x[frame_number]), float(self.location_y[frame_number]))
        self._patches[frame_number] = (
            crop_patch(image, center, self.radius, self.fill),
            patch_valid_area_ratio(image.shape, center, self.radius),
        )
        self._stats["patch_cache_misses"] += 1
        self._stats["peak_cached_patches"] = max(self._stats["peak_cached_patches"], len(self._patches))

    def _decode_through(self, frame_number: int) -> None:
        if self._capture is None:
            self._capture = cv2.VideoCapture(str(self.video))
            if not self._capture.isOpened():
                raise ValueError(f"无法打开视频: {self.video}")
            self._stats["video_open_count"] += 1
        while self._next_frame <= frame_number:
            readable, image = self._capture.read()
            if not readable:
                raise ValueError(
                    f"无法解码视频帧: current={self._next_frame}, source={self._next_frame}, target={frame_number}"
                )
            current = self._next_frame
            self._next_frame += 1
            self._stats["decoded_frame_count"] += 1
            _validate_image(image)
            if image.shape[:2] != (self.height, self.width):
                raise ValueError(f"视频第 {current} 帧分辨率不一致")
            if self._remaining_references[current] > 0 and self.quality[current] != QUALITY_IDS["missing"]:
                self._prepare(current, image)


def _nearest_frame_plan(timeline: np.ndarray, offsets: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros((len(timeline), len(offsets)), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=np.bool_)
    for column, offset in enumerate(offsets):
        targets = timeline + offset
        valid[:, column] = (targets >= timeline[0]) & (targets <= timeline[-1])
        selected = targets[valid[:, column]]
        right = np.minimum(np.searchsorted(timeline, selected, side="left"), len(timeline) - 1)
        left = np.maximum(right - 1, 0)
        choose_left = selected - timeline[left] <= timeline[right] - selected + 1e-12
        indices[valid[:, column], column] = np.where(choose_left, left, right)
    return indices, valid


def _timeline(values: Sequence[float] | np.ndarray, count: int | None = None) -> np.ndarray:
    timeline = np.asarray(values, dtype=np.float64)
    if timeline.ndim != 1 or not len(timeline) or not np.isfinite(timeline).all() or np.any(np.diff(timeline) <= 0):
        raise ValueError("timeline 必须是一维、有限且严格递增")
    if count is not None and timeline.shape != (count,):
        raise ValueError(f"视频时间轴帧数 {len(timeline)} 与轨迹帧数 {count} 不一致")
    return timeline


def _crop_bounds(center: Sequence[float], radius: int, width: int, height: int):
    center_x = _round_half_away_from_zero(float(center[0]))
    center_y = _round_half_away_from_zero(float(center[1]))
    left, top = center_x - radius, center_y - radius
    return left, top, max(left, 0), max(top, 0), min(left + 2 * radius, width), min(top + 2 * radius, height)


def _round_half_away_from_zero(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _fill(values: Sequence[int], dtype: np.dtype) -> np.ndarray:
    array = np.asarray(tuple(values))
    if array.shape != (3,) or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("fill 必须包含三个有限数值")
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        if np.any(array != np.floor(array)) or np.any(array < limits.min) or np.any(array > limits.max):
            raise ValueError(f"fill 无法转换为 {dtype}")
    return array.astype(dtype)


def _size(values: Sequence[int]) -> tuple[int, int]:
    size = tuple(values)
    if len(size) != 2 or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in size):
        raise ValueError("size 必须是两个正整数 [width,height]")
    return int(size[0]), int(size[1])


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("图像必须是 BGR uint8 HWC 三通道数组")


def _positive_dimension(value: float, name: str) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"视频{name}必须是有限正数")
    rounded = int(round(value))
    if rounded <= 0:
        raise ValueError(f"视频{name}取整后必须为正数")
    return rounded


def _positive_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
