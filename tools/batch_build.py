"""
batch_build.py - 批次將 pack 的 raw/ 圖片轉換為 APNG，輸出到 output/

使用方式:
    python tools/batch_build.py packs/pack_01_daily
    python tools/batch_build.py packs/pack_02_no_work
"""

import glob
import re
import os
import sys
import argparse
from pathlib import Path

# 允許從任意目錄呼叫，自動找到 build_stickers 模組
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
import build_stickers


def run_batch(pack_dir: str):
    pack_path = Path(pack_dir).resolve()
    raw_dir = pack_path / "raw"
    output_dir = pack_path / "output"

    if not raw_dir.exists():
        print(f"[ERROR] raw/ folder not found in: {pack_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    sprites = sorted(raw_dir.glob("scene_*.png"))
    if not sprites:
        print(f"[WARN] No scene_*.png files found in {raw_dir}")
        return

    for sprite in sprites:
        match = re.search(r"scene_(\d+)", sprite.name)
        if match:
            out_name = output_dir / f"{match.group(1)}.png"
            print(f"Processing {sprite.name} -> {out_name.name} ...")
            build_stickers.process_single_image(
                str(sprite), str(out_name), total_frames=5, duration_ms=200
            )

    print(f"\nDone! {len(sprites)} sticker(s) built to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch build animated LINE stickers for a pack"
    )
    parser.add_argument(
        "pack",
        help="Path to the pack folder (e.g. packs/pack_01_daily)",
    )
    args = parser.parse_args()
    run_batch(args.pack)
