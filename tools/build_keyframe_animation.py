"""
build_keyframe_animation.py - 多關鍵影格 (Keyframes) 動畫組裝工具

讀取指定目錄下的關鍵姿勢圖檔 (例如 pose_01.png, pose_02.png, pose_03.png, pose_04.png)，
自動產生流暢過渡中間畫格 (In-between Frames)，合成大動作連貫動畫，並輸出符合 LINE 官方標準的 16 格 4.0 秒 APNG 與預覽 GIF！
"""

import os
import sys
import glob
import re
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from apng import APNG

def remove_bg(img_bgra, tolerance=(28, 28, 28)):
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

def ensure_white_border(img_bgra, border_px=10):
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

def assemble_keyframes(keyframes_dir, output_apng_path, output_gif_path, total_frames=16, total_duration_s=4.0):
    keyframe_files = sorted(glob.glob(os.path.join(keyframes_dir, "pose_*.png")))
    if not keyframe_files:
        # Fallback to any png
        keyframe_files = sorted(glob.glob(os.path.join(keyframes_dir, "*.png")))

    if not keyframe_files:
        print(f"[ERROR] No keyframe images found in {keyframes_dir}")
        return

    print(f"Loaded {len(keyframe_files)} keyframe(s): {[os.path.basename(f) for f in keyframe_files]}")

    processed_keys = []
    # Target uniform size
    target_canvas_w = 320
    target_canvas_h = 270

    for kf_file in keyframe_files:
        raw = cv2.imread(kf_file, cv2.IMREAD_UNCHANGED)
        if raw is None: continue

        # Check alpha
        if raw.shape[2] == 3 or not np.any(raw[:, :, 3] < 250):
            raw = remove_bg(raw)
        raw = ensure_white_border(raw, border_px=10)

        # Crop tight content
        alpha = raw[:, :, 3]
        y_idx, x_idx = np.where(alpha > 10)
        if len(y_idx) > 0:
            raw_cropped = raw[min(y_idx):max(y_idx)+1, min(x_idx):max(x_idx)+1]
        else:
            raw_cropped = raw

        # Convert to PIL and fit nicely in 320x270
        pil_raw = Image.fromarray(cv2.cvtColor(raw_cropped, cv2.COLOR_BGRA2RGBA))
        cw, ch = pil_raw.size
        scale = min(320.0 / cw, 270.0 / ch)
        nw = int(round(cw * scale))
        nh = int(round(ch * scale))
        if nw % 2 != 0: nw -= 1
        if nh % 2 != 0: nh -= 1

        resized = pil_raw.resize((nw, nh), Image.Resampling.LANCZOS)
        
        # HD unsharp mask
        r_c, g_c, b_c, a_c = resized.split()
        rgb_c = Image.merge('RGB', (r_c, g_c, b_c)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        nr, ng, nb = rgb_c.split()
        sharpened = Image.merge('RGBA', (nr, ng, nb, a_c))

        canvas = Image.new("RGBA", (target_canvas_w, target_canvas_h), (0, 0, 0, 0))
        offset_x = (target_canvas_w - nw) // 2
        offset_y = (target_canvas_h - nh) // 2
        canvas.paste(sharpened, (offset_x, offset_y), sharpened)
        processed_keys.append(canvas)

    n_keys = len(processed_keys)
    if n_keys == 0:
        return

    # Build sequence of total_frames (e.g. 16 frames)
    # Distribute keys evenly around the loop
    frames_sequence = []
    frames_per_key = total_frames / float(n_keys)
    
    for i in range(total_frames):
        curr_key_idx = int(i / frames_per_key) % n_keys
        next_key_idx = (curr_key_idx + 1) % n_keys
        
        fraction = (i % frames_per_key) / float(frames_per_key)
        
        # Smooth ease in-out
        t = (1.0 - math.cos(fraction * math.pi)) / 2.0
        
        # Cross blend between keyframes
        img_a = processed_keys[curr_key_idx]
        img_b = processed_keys[next_key_idx]
        
        blended = Image.blend(img_a, img_b, t)
        
        # Add micro animation shake during high intensity frames
        if curr_key_idx in [1, 2]: # During crying / shaking phase
            shake_x = int(math.sin(i * 3.0) * 3)
            shake_y = int(math.cos(i * 4.0) * 2)
            shaken_canvas = Image.new("RGBA", (target_canvas_w, target_canvas_h), (0, 0, 0, 0))
            shaken_canvas.paste(blended, (shake_x, shake_y), blended)
            frames_sequence.append(shaken_canvas)
        else:
            frames_sequence.append(blended)

    # Frame duration in ms
    frame_ms = int((total_duration_s * 1000) / total_frames) # e.g. 4000ms / 16 = 250ms

    # Save APNG
    temp_files = []
    temp_dir = os.path.dirname(output_apng_path)
    for idx, f in enumerate(frames_sequence):
        tmp_name = os.path.join(temp_dir, f"temp_kf_{idx:02d}.png")
        qf = f.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    apng = APNG(num_plays=0)
    for tmp_name in temp_files:
        apng.append_file(tmp_name, delay=frame_ms, delay_den=1000)

    apng.save(output_apng_path)

    # Save GIF
    frames_sequence[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames_sequence[1:],
        duration=frame_ms,
        loop=0,
        disposal=2
    )

    for tmp_name in temp_files:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # Export contact sheet
    fw, fh = frames_sequence[0].size
    sheet = Image.new('RGBA', (fw * 4, fh * 4), (240, 240, 240, 255))
    for idx, f in enumerate(frames_sequence):
        r = idx // 4
        c = idx % 4
        sheet.paste(f, (c * fw, r * fh), f)
    sheet.save(os.path.join(temp_dir, "scene_15_keyframe_sheet.png"))

    print(f"[SUCCESS] Built 16-frame APNG ({total_duration_s}s loop): {output_apng_path} ({os.path.getsize(output_apng_path)/1024:.1f} KB)")
    print(f"[SUCCESS] Built preview GIF: {output_gif_path} ({os.path.getsize(output_gif_path)/1024:.1f} KB)")

if __name__ == "__main__":
    kdir = sys.argv[1] if len(sys.argv) > 1 else "packs/pack_02_no_work/animated_keyframes/scene_15"
    out_apng = sys.argv[2] if len(sys.argv) > 2 else "packs/pack_02_no_work/animated_demo/scene_15_keyframe_story.png"
    out_gif = sys.argv[3] if len(sys.argv) > 3 else "packs/pack_02_no_work/animated_demo/scene_15_keyframe_story_preview.gif"
    assemble_keyframes(kdir, out_apng, out_gif)
