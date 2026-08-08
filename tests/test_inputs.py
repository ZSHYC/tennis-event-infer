from __future__ import annotations

import numpy as np
import cv2
import pytest
import torch
from types import SimpleNamespace

import tennis_event_infer.video as video_module
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
    letterbox_image,
    locate_patch_center_arrays,
    nearest_frame_indices,
    patch_valid_area_ratio,
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


def test_packet_timeline_sorts_pts_and_matches_opencv_float_order():
    payload = {
        "packets": [{"pts": 220}, {"pts": 0}, {"pts": 100}],
        "streams": [{"time_base": "1/600"}],
    }

    timeline = video_module._packet_timeline(payload, expected_count=3)

    assert timeline.dtype == np.float64
    assert timeline.tolist() == [0.0, 0.16666666666666669, 0.3666666666666667]


@pytest.mark.parametrize(
    ("payload", "expected_count"),
    [
        ({"packets": [{"pts": 0}], "streams": [{"time_base": "0/1"}]}, 1),
        ({"packets": [{"pts": 0}, {"pts": 0}], "streams": [{"time_base": "1/10"}]}, 2),
        ({"packets": [{}], "streams": [{"time_base": "1/10"}]}, 1),
        ({"packets": [{"pts": -1}], "streams": [{"time_base": "1/10"}]}, 1),
        ({"packets": [{"pts": "0"}], "streams": [{"time_base": "1/10"}]}, 1),
        ({"packets": [{"pts": 0}], "streams": [{"time_base": "1/10"}]}, 2),
    ],
)
def test_packet_timeline_rejects_invalid_probe_data(payload, expected_count):
    with pytest.raises(ValueError):
        video_module._packet_timeline(payload, expected_count=expected_count)


def test_scan_video_uses_packet_timeline_without_grabbing_frames(tmp_path, monkeypatch):
    path = _video(tmp_path)
    real_capture = cv2.VideoCapture
    grabs = 0

    class CountingCapture:
        def __init__(self, source):
            self.capture = real_capture(source)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def grab(self):
            nonlocal grabs
            grabs += 1
            return self.capture.grab()

    timeline = np.arange(8, dtype=np.float64) / 10
    monkeypatch.setattr(cv2, "VideoCapture", CountingCapture)
    monkeypatch.setattr(video_module, "_ffprobe_timeline", lambda _video, _count: timeline)

    info = scan_video(path)

    assert grabs == 0
    assert info.timestamp_source == "pts"
    np.testing.assert_array_equal(info.timeline, timeline)


def test_scan_video_falls_back_when_packet_timeline_is_unavailable(tmp_path, monkeypatch):
    path = _video(tmp_path)
    monkeypatch.setattr(video_module, "_ffprobe_timeline", lambda _video, _count: None)

    info = scan_video(path)

    assert info.frame_count == 8
    assert info.timestamp_source in {"pts", "fps_fallback"}
    assert np.all(np.diff(info.timeline) > 0)


def test_ffprobe_timeline_missing_binary_reports_install_command(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(video_module.shutil, "which", lambda _name: None)

    assert video_module._ffprobe_timeline(tmp_path / "video.mov", 1) is None
    assert "sudo apt-get install -y ffmpeg" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        video_module.subprocess.TimeoutExpired("ffprobe", 30),
        video_module.subprocess.CalledProcessError(1, ["ffprobe"]),
    ],
)
def test_ffprobe_timeline_command_failure_returns_none(tmp_path, monkeypatch, error):
    monkeypatch.setattr(video_module.shutil, "which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(video_module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    assert video_module._ffprobe_timeline(tmp_path / "video.mov", 1) is None


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


def _prepare_only_reader(*, center, radius=2, size=(4, 4)):
    reader = object.__new__(PatchReader)
    reader.location_x = np.asarray([center[0]], dtype=np.float32)
    reader.location_y = np.asarray([center[1]], dtype=np.float32)
    reader.radius = radius
    reader.fill = (114, 114, 114)
    reader.size = size
    reader._patches = {}
    reader._stats = {"patch_cache_misses": 0, "peak_cached_patches": 0}
    return reader


def test_patch_reader_prepare_uses_view_for_fully_inside_patch(monkeypatch):
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    center, radius = (4.0, 4.0), 2
    expected_patch = crop_patch(image, center, radius, (114, 114, 114))
    expected = np.moveaxis(
        letterbox_image(expected_patch, (4, 4), (114, 114, 114))[..., ::-1], -1, 0
    ).copy()
    reader = _prepare_only_reader(center=center, radius=radius)
    monkeypatch.setattr(
        video_module,
        "crop_patch",
        lambda *args, **kwargs: pytest.fail("fully inside must not copy"),
    )

    reader._prepare(0, image)

    actual, area = reader._patches[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.shape == (3, 4, 4)
    assert area == 1.0


def test_patch_reader_prepare_keeps_boundary_fill_semantics(monkeypatch):
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    center, radius = (0.0, 0.0), 2
    expected_patch = crop_patch(image, center, radius, (114, 114, 114))
    expected = np.moveaxis(
        letterbox_image(expected_patch, (4, 4), (114, 114, 114))[..., ::-1], -1, 0
    ).copy()
    calls = 0
    real_crop = crop_patch

    def counted_crop(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_crop(*args, **kwargs)

    reader = _prepare_only_reader(center=center, radius=radius)
    monkeypatch.setattr(video_module, "crop_patch", counted_crop)

    reader._prepare(0, image)

    np.testing.assert_array_equal(reader._patches[0][0], expected)
    assert calls == 1
    assert reader.stats == {"patch_cache_misses": 1, "peak_cached_patches": 1}


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


def _reference_sequences(path, frames, info, contract):
    capture = cv2.VideoCapture(str(path))
    images = []
    while True:
        readable, image = capture.read()
        if not readable:
            break
        images.append(image)
    capture.release()
    x, y, quality, distance = locate_patch_center_arrays(
        frames, max_gap_seconds=contract.max_gap_seconds, timeline=info.timeline
    )
    radius = max(1, int(round(contract.radius_ratio * min(info.width, info.height))))
    expected = []
    for center in range(len(frames)):
        indices, valid = nearest_frame_indices(info.timeline, center=center, offsets=contract.offsets_seconds)
        pixels = np.zeros((5, 3, contract.size[1], contract.size[0]), dtype=np.uint8)
        mask = np.zeros(5, dtype=np.bool_)
        qualities = np.zeros(5, dtype=np.int64)
        continuous = np.zeros((5, 3), dtype=np.float32)
        for output_index, source in enumerate(indices):
            if not valid[output_index]:
                continue
            qualities[output_index] = quality[source]
            if quality[source] == 0:
                continue
            center_xy = (float(x[source]), float(y[source]))
            patch = crop_patch(images[source], center_xy, radius, contract.fill)
            prepared = letterbox_image(patch, contract.size, contract.fill)
            pixels[output_index] = np.moveaxis(prepared[..., ::-1], -1, 0)
            mask[output_index] = True
            continuous[output_index] = (
                abs(float(info.timeline[source]) - source / info.fps),
                float(distance[source]),
                patch_valid_area_ratio(images[source].shape, center_xy, radius),
            )
        expected.append(
            {
                "patch_pixels": torch.from_numpy(pixels),
                "patch_mask": torch.from_numpy(mask),
                "patch_quality": torch.from_numpy(qualities),
                "patch_quality_continuous": torch.from_numpy(continuous),
            }
        )
    return expected


def test_patch_reader_matches_reference_with_one_forward_decode(tmp_path):
    path = _video(tmp_path)
    before = {item.name for item in tmp_path.iterdir()}
    info = scan_video(path)
    frames = _frames()
    contract = _patch_contract()
    expected = _reference_sequences(path, frames, info, contract)
    reader = PatchReader(path, frames, info.timeline, contract, cache_size=5)
    results = [reader.sequence(center) for center in range(len(frames))]
    stats = reader.stats
    reader.close()
    for result, reference in zip(results, expected):
        for name in reference:
            assert torch.equal(result[name], reference[name]), name
    assert stats["video_open_count"] == 1
    assert stats["decoded_frame_count"] <= len(frames)
    assert {item.name for item in tmp_path.iterdir()} == before


def test_patch_reader_rejects_boolean_center(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    with pytest.raises(ValueError, match="center=False"):
        reader.sequence(False)
    assert reader.stats["backward_request_count"] == 1
    assert reader.stats["video_open_count"] == 0
    reader.close()


def test_patch_reader_duplicate_sources_use_one_cache_slot_and_release_at_end(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    contract = _patch_contract()
    contract.offsets_seconds = (0.0,) * 5
    frames = _frames()
    expected = _reference_sequences(path, frames, info, contract)
    reader = PatchReader(path, frames, info.timeline, contract, cache_size=1)
    results = [reader.sequence(center) for center in range(len(frames))]
    for result, reference in zip(results, expected):
        for name in reference:
            assert torch.equal(result[name], reference[name]), name
    assert reader.stats["peak_cached_patches"] == 1
    assert reader.stats["decoded_frame_count"] == len(frames)
    assert reader._patches == {}
    reader.close()


def test_patch_reader_prepares_each_source_patch_once(tmp_path, monkeypatch):
    path = _video(tmp_path)
    info = scan_video(path)
    contract = _patch_contract()
    contract.offsets_seconds = (0.0,) * 5
    calls = []
    real_letterbox = video_module.letterbox_image

    def count_letterbox(*args, **kwargs):
        calls.append(None)
        return real_letterbox(*args, **kwargs)

    monkeypatch.setattr(video_module, "letterbox_image", count_letterbox)
    reader = PatchReader(path, _frames(), info.timeline, contract, cache_size=1)

    for center in range(len(_frames())):
        reader.sequence(center)

    assert len(calls) == len(_frames())
    reader.close()


def test_patch_reader_preserves_boundary_and_missing_outputs(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    frames = [
        TrajectoryFrame(index, index in {0, 7}, 4.0 if index in {0, 7} else None, 3.0 if index in {0, 7} else None, 16, 8)
        for index in range(8)
    ]
    contract = _patch_contract()
    contract.max_gap_seconds = 0.01
    expected = _reference_sequences(path, frames, info, contract)
    reader = PatchReader(path, frames, info.timeline, contract, cache_size=2)
    results = [reader.sequence(center) for center in range(len(frames))]
    for result, reference in zip(results, expected):
        for name in reference:
            assert torch.equal(result[name], reference[name]), name
    assert any(not result["patch_mask"].all() for result in results)
    assert reader._patches == {}
    reader.close()


def test_patch_reader_rejects_insufficient_cache_before_decode(tmp_path, monkeypatch):
    path = _video(tmp_path, count=16)
    info = scan_video(path)
    frames = _frames(16)
    contract = _patch_contract()
    real_capture = cv2.VideoCapture
    reads = 0

    class CountingCapture:
        def __init__(self, source):
            self.capture = real_capture(source)

        def __getattr__(self, name):
            return getattr(self.capture, name)

        def read(self):
            nonlocal reads
            reads += 1
            return self.capture.read()

    monkeypatch.setattr(cv2, "VideoCapture", CountingCapture)
    with pytest.raises(ValueError, match=r"峰值 .* cache_size=1.*尚未开始视频解码"):
        PatchReader(path, frames, info.timeline, contract, cache_size=1)
    assert reads == 0


def test_patch_reader_rejects_out_of_order_without_reopening(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    reader.sequence(0)
    before = reader.stats
    with pytest.raises(ValueError, match=r"center=0.*expected=1.*source="):
        reader.sequence(0)
    after = reader.stats
    assert after["backward_request_count"] == before["backward_request_count"] + 1
    assert after["video_open_count"] == before["video_open_count"] == 1
    assert after["decoded_frame_count"] == before["decoded_frame_count"]
    reader.close()


def test_patch_reader_stats_are_copied(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    stats = reader.stats
    stats["video_open_count"] = 99
    assert reader.stats["video_open_count"] == 0
    reader.close()
    assert reader.stats["video_open_count"] == 0


def test_patch_reader_close_is_idempotent_after_release_error(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    reader.sequence(0)
    reader._capture.release()

    class FailingCapture:
        def release(self):
            raise RuntimeError("release failed")

    reader._capture = FailingCapture()
    with pytest.raises(RuntimeError, match="release failed"):
        reader.close()
    assert reader._capture is None
    assert reader._patches == {}
    assert reader._remaining_references is None
    reader.close()


def test_patch_reader_decode_error_reports_current_source_and_target(tmp_path, monkeypatch):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)

    class UnreadableCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: UnreadableCapture())
    with pytest.raises(ValueError, match=r"current=0, source=0, target=2"):
        reader.sequence(0)
    reader.close()


def test_patch_reader_sequence_after_close_has_stable_error(tmp_path):
    path = _video(tmp_path)
    info = scan_video(path)
    reader = PatchReader(path, _frames(), info.timeline, _patch_contract(), cache_size=5)
    reader.close()
    with pytest.raises(ValueError, match=r"^PatchReader 已关闭$"):
        reader.sequence(0)
