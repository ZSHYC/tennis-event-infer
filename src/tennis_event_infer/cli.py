from __future__ import annotations

import argparse
from pathlib import Path
import time

from .pipeline import run_inference


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="使用视频、TrackNetV5 CSV 和 deployment checkpoint 检测网球事件。")
    result.add_argument("--video", required=True, type=Path)
    result.add_argument("--trajectory", required=True, type=Path)
    result.add_argument("--checkpoint", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--batch-size", type=int, default=128)
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    started = time.monotonic()
    try:
        summary = run_inference(
            video=args.video,
            trajectory=args.trajectory,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (EOFError, FileNotFoundError, OSError, ValueError) as exc:
        argument_parser.error(str(exc))
    summary["elapsed_seconds"] = time.monotonic() - started
    for name in ("device", "frames", "events", "elapsed_seconds", "events_json", "events_csv"):
        print(f"{name}: {summary[name]}")


if __name__ == "__main__":
    main()
