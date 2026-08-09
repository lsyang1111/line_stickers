"""
build_static_pack.py - 批次處理 raw/ 圖片，生成符合 LINE 官方規範的「一般/靜態貼圖」PNG 與上傳 ZIP 包

規範：
- 貼圖本體 (01.png ~ 16.png): 最大 W 370 x H 320 px (偶數)，RGB/RGBA 透明背景 PNG，單檔 < 1MB
- 主要圖片 (main.png): 固定 W 240 x H 240 px (偶數)，RGB/RGBA 透明背景 PNG
- 標籤圖片 (tab.png): 固定 W 96 x H 74 px (偶數)，RGB/RGBA 透明背景 PNG
- ZIP 壓縮包直接包含 18 個圖檔，無多餘子目錄
"""

import os
import re
import glob
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def remove_background(img_bgra, tolerance=(30, 30, 30)):
    h, w = img_bgra.shape[:2]
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)
    bgr = img_bgra[:, :, :3].copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    perimeter = []
    for x in range(0, w, 4):
        perimeter.append((x, 0))
        perimeter.append((x, h - 1))
    for y in range(0, h, 4):
        perimeter.append((0, y))
        perimeter.append((w - 1, y))

    for pt in perimeter:
        if mask[pt[1] + 1, pt[0] + 1] == 0:
            bg_color = bgr[pt[1], pt[0]]
            if int(bg_color[0]) + int(bg_color[1]) + int(bg_color[2]) > 280:
                cv2.floodFill(bgr, mask, pt, (0, 255, 0), tolerance, tolerance, cv2.FLOODFILL_FIXED_RANGE)

    bg_mask = mask[1:-1, 1:-1]
    fg_mask = (bg_mask == 0).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(fg_mask, (3, 3), 0)
    img_bgra[:, :, 3] = alpha
    return img_bgra

def ensure_white_diecut_border(img_bgra, border_px=12):
    alpha = img_bgra[:, :, 3]
    binary = (alpha > 20).astype(np.uint8) * 255

    k_size = border_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_mask = cv2.dilate(binary, kernel)
    dilated_alpha = cv2.GaussianBlur(dilated_mask, (5, 5), 0)

    h, w = img_bgra.shape[:2]
    white_layer = np.full((h, w, 4), 255, dtype=np.uint8)
    white_layer[:, :, 3] = dilated_alpha

    fg_pil = Image.fromarray(cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA))
    bg_pil = Image.fromarray(cv2.cvtColor(white_layer, cv2.COLOR_BGRA2RGBA))
    combined = Image.alpha_composite(bg_pil, fg_pil)
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGBA2BGRA)

def process_raw_to_rgba_pil(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot load image: {input_path}")

    has_alpha = False
    if len(img.shape) == 3 and img.shape[2] == 4:
        if np.any(img[:, :, 3] < 250):
            has_alpha = True
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if not has_alpha:
        img = remove_background(img)

    img = ensure_white_diecut_border(img, border_px=10)

    alpha = img[:, :, 3]
    y_idx, x_idx = np.where(alpha > 10)
    if len(y_idx) == 0:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
    y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
    cropped = img[y_min:y_max + 1, x_min:x_max + 1]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))

def create_static_sticker(cropped_pil, target_w=370, target_h=320, padding=12):
    max_w = target_w - padding * 2
    max_h = target_h - padding * 2

    cw, ch = cropped_pil.size
    scale = min(max_w / cw, max_h / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))

    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    resized = cropped_pil.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas

def create_main_image(cropped_pil, target_size=240, padding=10):
    max_content = target_size - padding * 2
    cw, ch = cropped_pil.size
    scale = min(max_content / cw, max_content / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))

    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    resized = cropped_pil.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - nw) // 2
    offset_y = (target_size - nh) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas

def create_tab_image(cropped_pil, target_w=96, target_h=74, padding=4):
    max_w = target_w - padding * 2
    max_h = target_h - padding * 2
    cw, ch = cropped_pil.size
    scale = min(max_w / cw, max_h / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))

    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    resized = cropped_pil.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    canvas.paste(resized, (offset_x, offset_y), resized)
    return canvas

def build_pack_static(pack_dir):
    pack_path = Path(pack_dir).resolve()
    raw_dir = pack_path / "raw"
    out_dir = pack_path / "output_static"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("scene_*.png"))
    print(f"Found {len(raw_files)} raw scenes in {raw_dir}")

    processed_stickers = {}
    for raw_file in raw_files:
        match = re.search(r"scene_(\d+)", raw_file.name)
        if not match:
            continue
        idx = int(match.group(1))
        print(f"Processing scene_{idx:02d} -> {idx:02d}.png ...")
        cropped_pil = process_raw_to_rgba_pil(str(raw_file))
        sticker_canvas = create_static_sticker(cropped_pil, 370, 320, padding=14)
        out_file = out_dir / f"{idx:02d}.png"
        sticker_canvas.save(str(out_file), "PNG", optimize=True)
        processed_stickers[idx] = cropped_pil

    first_idx = min(processed_stickers.keys()) if processed_stickers else 1
    main_canvas = create_main_image(processed_stickers[first_idx], 240, padding=10)
    main_file = out_dir / "main.png"
    main_canvas.save(str(main_file), "PNG", optimize=True)
    print(f"Generated {main_file.name} (240x240 PNG)")

    tab_canvas = create_tab_image(processed_stickers[first_idx], 96, 74, padding=4)
    tab_file = out_dir / "tab.png"
    tab_canvas.save(str(tab_file), "PNG", optimize=True)
    print(f"Generated {tab_file.name} (96x74 PNG)")

    zip_path = pack_path / "pack_02_stickers_upload.zip"
    static_zip_path = pack_path / "pack_02_static_upload.zip"
    
    for zp in [zip_path, static_zip_path]:
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out_dir.glob("*.png")):
                z.write(f, arcname=f.name)
        print(f"Created upload ZIP: {zp} ({zp.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    import sys
    pack_dir = sys.argv[1] if len(sys.argv) > 1 else "packs/pack_02_no_work"
    build_pack_static(pack_dir)
