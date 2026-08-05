from __future__ import annotations

import numpy as np
import cv2
import pytest
import torch
from types import SimpleNamespace

from tennis_event_infer.trajectory import (
    FEATURE_NAMES,
    FeatureNormalizer,
    load_v5_csv,
    make_trajectory_windows,
)
from tennis_event_infer.types import TrajectoryFrame
from tennis_event_infer.video import (
    PatchReader,
    crop_patch,
    locate_patch_center_arrays,
    nearest_frame_indices,
    scan_video,
)


def _write(tmp_path, text: str):
    path = tmp_path / "track.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_v5_requires_complete_zero_based_frames(tmp_path):
    path = _write(
        tmp_path,
        "frame_number,detected,x_orig,y_orig,width,height\n"
        "0,1,4,3,16,8\n"
        "2,0,,,16,8\n",
    )
    with pytest.raises(ValueError, match="0..N-1"):
        load_v5_csv(path)


def test_load_v5_ignores_extra_columns(tmp_path):
    path = _write(
        tmp_path,
        "frame_number,detected,x_orig,y_orig,width,height,conf,heatmap_peak\n"
        "0,1,4,3,16,8,0.9,1.0\n"
        "1,0,0,0,16,8,0.0,0.0\n",
    )
    frames = load_v5_csv(path)
    assert [(f.frame_number, f.detected, f.x, f.y) for f in frames] == [
        (0, True, 4.0, 3.0),
        (1, False, None, None),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "frame_number,detected,x_orig,width,height\n0,1,4,16,8\n",
            "缺少字段",
        ),
        (
            "frame_number,detected,x_orig,y_orig,width,height\n0,1,4,3,16,8\n0,0,,,16,8\n",
            "0..N-1",
        ),
        (
            "frame_number,detected,x_orig,y_orig,width,height\n0,2,4,3,16,8\n",
            "detected",
        ),
        (
            "frame_number,detected,x_orig,y_orig,width,height\n0,1,16,3,16,8\n",
            "画面范围",
        ),
        (
            "frame_number,detected,x_orig,y_orig,width,height\n0,1,4,3,16,8\n1,0,,,32,8\n",
            "分辨率",
        ),
    ],
)
def test_load_v5_rejects_invalid_rows(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_v5_csv(_write(tmp_path, text))


def test_trajectory_windows_match_frozen_feature_schema():
    frames = [
        TrajectoryFrame(0, True, 2.0, 2.0, 10, 10),
        TrajectoryFrame(1, False, None, None, 10, 10),
        TrajectoryFrame(2, True, 4.0, 2.0, 10, 10),
    ]
    windows = make_trajectory_windows(frames, radius=1, fps=10.0, max_gap_seconds=0.2)
    assert windows.shape == (3, 3, len(FEATURE_NAMES))
    assert windows.dtype == np.float32
    np.testing.assert_allclose(
        windows[1],
        [
            [0.2, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.4, 0.2, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0, 1.0],
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    assert windows[0, 0, FEATURE_NAMES.index("valid")] == 0.0


def test_feature_normalizer_uses_checkpoint_statistics_without_fit():
    contract = {
        "type": "robust",
        "feature_names": ["x_norm", "speed"],
        "medians": [0.2, 1.0],
        "scales": [0.1, 0.5],
        "clip": 2.0,
        "mask_feature": "detected",
    }
    values = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    values[:, FEATURE_NAMES.index("x_norm")] = [0.5, 0.5]
    values[:, FEATURE_NAMES.index("speed")] = [3.0, 3.0]
    values[:, FEATURE_NAMES.index("detected")] = [1.0, 0.0]
    transformed = FeatureNormalizer.from_contract(contract).transform(values)
    assert transformed[0, FEATURE_NAMES.index("x_norm")] == 2.0
    assert transformed[0, FEATURE_NAMES.index("speed")] == 2.0
    assert transformed[1, FEATURE_NAMES.index("x_norm")] == 0.0
    assert transformed[1, FEATURE_NAMES.index("speed")] == 0.0
    assert not hasattr(FeatureNormalizer, "fit")


@pytest.mark.parametrize(
    "change",
    [
        {"type": "standard"},
        {"feature_names": ["unknown"]},
        {"medians": [0.0]},
        {"scales": [0.0, 1.0]},
        {"clip": float("nan")},
        {"mask_feature": "unknown"},
    ],
)
def test_feature_normalizer_rejects_invalid_contract(change):
    contract = {
        "type": "robust",
        "feature_names": ["x_norm", "speed"],
        "medians": [0.2, 1.0],
        "scales": [0.1, 0.5],
        "clip": 2.0,
        "mask_feature": "detected",
    }
    contract.update(change)
    with pytest.raises(ValueError):
        FeatureNormalizer.from_contract(contract)


def _video(tmp_path, count: int = 8, size: tuple[int, int] = (16, 8)):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, size)
    assert writer.isOpened()
    for index in range(count):
        image = np.full((size[1], size[0], 3), (index, index + 1, index + 2), dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


def _frames(count: int = 8, size: tuple[int, int] = (16, 8)):
    return [TrajectoryFrame(index, True, 4.0 + index, 3.0, size[0], size[1]) for index in range(count)]


def _patch_contract():
    return SimpleNamespace(
        size=(8, 8),
        fill=(114, 114, 114),
        radius_ratio=0.25,
        max_gap_seconds=0.2,
        offsets_seconds=(-0.2, -0.1, 0.0, 0.1, 0.2),
    )


def test_scan_video_and_patch_reader_validate_complete_alignment(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    assert (info.frame_count, info.width, info.height) == (8, 16, 8)
    assert info.timeline.shape == (8,)
    assert np.all(np.diff(info.timeline) > 0)

    with pytest.raises(ValueError, match="帧数"):
        PatchReader(path, _frames(7), info.timeline, _patch_contract())
    wrong_size = _frames(size=(32, 8))
    with pytest.raises(ValueError, match="分辨率"):
        PatchReader(path, wrong_size, info.timeline, _patch_contract())


def test_nearest_frame_breaks_ties_to_earlier_frame():
    times = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    indices, valid = nearest_frame_indices(times, center=1, offsets=(-0.05, 0.05))
    assert indices.tolist() == [0, 1]
    assert valid.tolist() == [True, True]


def test_crop_patch_fills_outside_image():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    patch = crop_patch(image, center=(0.0, 0.0), radius=2, fill=(114, 114, 114))
    assert patch.shape == (4, 4, 3)
    assert tuple(patch[0, 0]) == (114, 114, 114)
    assert tuple(patch[2, 2]) == (0, 0, 0)


def test_patch_locations_interpolate_only_within_time_gap():
    frames = [
        TrajectoryFrame(0, True, 2.0, 3.0, 16, 8),
        TrajectoryFrame(1, False, None, None, 16, 8),
        TrajectoryFrame(2, True, 6.0, 3.0, 16, 8),
        TrajectoryFrame(3, False, None, None, 16, 8),
    ]
    x, y, quality, distance = locate_patch_center_arrays(
        frames, max_gap_seconds=0.11, timeline=np.arange(4, dtype=np.float64) / 10
    )
    assert (x[1], y[1], quality[1]) == (4.0, 3.0, 2)
    assert distance[1] == pytest.approx(0.1)
    assert (x[3], y[3], quality[3]) == (6.0, 3.0, 1)


def test_patch_reader_returns_five_online_patches_without_files(tmp_path):
    path = _video(tmp_path)
    before = {item.name for item in tmp_path.iterdir()}
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    result = reader.sequence(3)
    reader.close()
    assert result["patch_pixels"].shape == (5, 3, 8, 8)
    assert result["patch_pixels"].dtype == torch.uint8
    assert result["patch_mask"].dtype == torch.bool
    assert result["patch_quality"].dtype == torch.long
    assert result["patch_quality_continuous"].dtype == torch.float32
    assert result["patch_mask"].tolist() == [True] * 5
    assert {item.name for item in tmp_path.iterdir()} == before
