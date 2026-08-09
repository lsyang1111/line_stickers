"""
build_static_pack.py - 批次處理 raw/ 圖片，生成符合 LINE 官方規範的高畫質「一般/靜態貼圖」PNG 與上傳 ZIP 包

高畫質 (HD) 最佳化策略：
1. 超取樣白邊處理 (Super-Sampled Die-Cut Border)：在原圖最高解析度下進行邊緣膨脹與次像素羽化，縮小後邊緣極度平滑無鋸齒。
2. 智慧銳化修復 (Smart Unsharp Mask)：針對縮圖後損失的高頻毛髮細節、貓咪眼神、鬍鬚與文字輪廓進行微米級細節銳化，使在手機視網膜螢幕上清晰立體。
3. 微對比度與色彩飽和調校 (Micro-Contrast Tuning)：微調 4~5% 對比與鮮豔度，在 LINE 深色/淺色對話背景下皆呈現飽滿質感。
4. 最大顯示面積最佳化 (Max Visual Area)：精準控制邊距 (8px 安全邊界)，讓貼圖在對話框內視覺最大化。
5. 32-bit Lossless Truecolor RGBA：不進行失真調色板壓縮，保留 100% 原生漸層與光影。

規範標準：
- 貼圖本體 (01.png ~ 16.png): 最大 W 370 x H 320 px (偶數)，RGB/RGBA 透明背景 PNG，單檔 < 1MB
- 主要圖片 (main.png): 固定 W 240 x H 240 px (偶數)，RGB/RGBA 透明背景 PNG
- 標籤圖片 (tab.png): 固定 W 96 x H 74 px (偶數)，RGB/RGBA 透明背景 PNG
"""

import os
import re
import glob
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def remove_background(img_bgra, tolerance=(30, 30, 30)):
    h, w = img_bgra.shape[:2]
    if img_bgra.shape[2] == 3:
        img_bgra = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2BGRA)
    bgr = img_bgra[:, :, :3].copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    perimeter = []
    step = max(4, min(h, w) // 250)
    for x in range(0, w, step):
        perimeter.append((x, 0))
        perimeter.append((x, h - 1))
    for y in range(0, h, step):
        perimeter.append((0, y))
        perimeter.append((w - 1, y))

    for pt in perimeter:
        if mask[pt[1] + 1, pt[0] + 1] == 0:
            bg_color = bgr[pt[1], pt[0]]
            if int(bg_color[0]) + int(bg_color[1]) + int(bg_color[2]) > 280:
                cv2.floodFill(bgr, mask, pt, (0, 255, 0), tolerance, tolerance, cv2.FLOODFILL_FIXED_RANGE)

    bg_mask = mask[1:-1, 1:-1]
    fg_mask = (bg_mask == 0).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    img_bgra[:, :, 3] = alpha
    return img_bgra

def ensure_white_diecut_border(img_bgra, border_px_ratio=0.012):
    """
    在原始高解析度下進行立體白框膨脹，使縮圖後的白色剪紙外框呈現向量級平滑抗鋸齒
    """
    h, w = img_bgra.shape[:2]
    border_px = max(6, int(min(h, w) * border_px_ratio))

    alpha = img_bgra[:, :, 3]
    binary = (alpha > 20).astype(np.uint8) * 255

    k_size = border_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_mask = cv2.dilate(binary, kernel)
    dilated_alpha = cv2.GaussianBlur(dilated_mask, (7, 7), 0)

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

    img = ensure_white_diecut_border(img, border_px_ratio=0.012)

    alpha = img[:, :, 3]
    y_idx, x_idx = np.where(alpha > 10)
    if len(y_idx) == 0:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
    y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
    cropped = img[y_min:y_max + 1, x_min:x_max + 1]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))

def enhance_and_sharpen_hd(pil_img):
    """
    高品質縮圖後的高頻毛髮、眼神與文字邊緣細節強化
    """
    r, g, b, a = pil_img.split()
    rgb = Image.merge('RGB', (r, g, b))
    
    # 1. 精細銳化：恢復因縮小而微糊的貓毛與鬍鬚
    sharpened_rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=2))
    
    # 2. 視網膜螢幕微對比度增強
    enhanced_rgb = ImageEnhance.Contrast(sharpened_rgb).enhance(1.04)
    enhanced_rgb = ImageEnhance.Color(enhanced_rgb).enhance(1.05)
    
    nr, ng, nb = enhanced_rgb.split()
    return Image.merge('RGBA', (nr, ng, nb, a))

def create_static_sticker(cropped_pil, target_w=370, target_h=320, padding=8):
    max_w = target_w - padding * 2
    max_h = target_h - padding * 2

    cw, ch = cropped_pil.size
    scale = min(max_w / cw, max_h / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))

    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    # 使用 Lanczos 頂級插值縮放
    resized = cropped_pil.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # HD 銳化與微對比增強
    hd_resized = enhance_and_sharpen_hd(resized)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    canvas.paste(hd_resized, (offset_x, offset_y), hd_resized)
    return canvas

def create_main_image(cropped_pil, target_size=240, padding=8):
    max_content = target_size - padding * 2
    cw, ch = cropped_pil.size
    scale = min(max_content / cw, max_content / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))

    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    resized = cropped_pil.resize((nw, nh), Image.Resampling.LANCZOS)
    hd_resized = enhance_and_sharpen_hd(resized)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - nw) // 2
    offset_y = (target_size - nh) // 2
    canvas.paste(hd_resized, (offset_x, offset_y), hd_resized)
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
    hd_resized = enhance_and_sharpen_hd(resized)

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    canvas.paste(hd_resized, (offset_x, offset_y), hd_resized)
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
        print(f"Processing HD scene_{idx:02d} -> {idx:02d}.png ...")
        cropped_pil = process_raw_to_rgba_pil(str(raw_file))
        sticker_canvas = create_static_sticker(cropped_pil, 370, 320, padding=8)
        out_file = out_dir / f"{idx:02d}.png"
        sticker_canvas.save(str(out_file), "PNG", optimize=True)
        processed_stickers[idx] = cropped_pil

    first_idx = min(processed_stickers.keys()) if processed_stickers else 1
    main_canvas = create_main_image(processed_stickers[first_idx], 240, padding=8)
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
        print(f"Created HD upload ZIP: {zp} ({zp.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    import sys
    pack_dir = sys.argv[1] if len(sys.argv) > 1 else "packs/pack_02_no_work"
    build_pack_static(pack_dir)
