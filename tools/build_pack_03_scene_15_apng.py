"""
build_pack_03_scene_15_apng.py - 為 pack_03_no_work 建立符合 LINE 官方 APNG 標準的 4 幕大動作劇情動畫

4 段大動作關鍵影格：
Pose A (pose_04.png): 捧手機看暴跌圖表，眼神呆滯放空 (震驚期)
Pose B (pose_01.png): 抱著手機癱坐，委屈用爪子抹眼淚 (悲傷期)
Pose C (pose_03.png): 雙爪抱頭仰天大噴兩道粗大藍色瀑布淚！(痛哭爆發期)
Pose D (pose_02.png): 撲上前雙爪死命搖晃欄杆，大喊 LET ME OUT! (求救期)
回歸循環至 Pose A

技術指標：
- 畫格數：10 格
- 播放時長：2.0 秒循環 (每格 200ms) 或 3.0 秒 (每格 300ms)
- 檔案大小：< 300 KB (嚴格符合 LINE Creators Market 規範)
- 尺寸：320 x 270 px (偶數、透明背景、立體抗鋸齒白邊)
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from apng import APNG

def remove_background_clean(img_bgr, tolerance=(25, 25, 25)):
    h, w = img_bgr.shape[:2]
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgr = img_bgr.copy()
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
            if int(bg_color[0]) + int(bg_color[1]) + int(bg_color[2]) > 300:
                cv2.floodFill(bgr, mask, pt, (0, 255, 0), tolerance, tolerance, cv2.FLOODFILL_FIXED_RANGE)

    bg_mask = mask[1:-1, 1:-1]
    fg_mask = (bg_mask == 0).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(fg_mask, (3, 3), 0)
    img_bgra[:, :, 3] = alpha
    return img_bgra

def ensure_white_diecut(img_bgra, border_px=6):
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

def prepare_clean_keyframe(file_path, target_w=320, target_h=270):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read {file_path}")

    bgra = remove_background_clean(img)
    bgra = ensure_white_diecut(bgra, border_px=6)

    alpha = bgra[:, :, 3]
    y_idx, x_idx = np.where(alpha > 10)
    if len(y_idx) > 0:
        cropped = bgra[min(y_idx):max(y_idx)+1, min(x_idx):max(x_idx)+1]
    else:
        cropped = bgra

    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))
    cw, ch = pil_img.size
    scale = min(320.0 / cw, 270.0 / ch)
    nw = int(round(cw * scale))
    nh = int(round(ch * scale))
    if nw % 2 != 0: nw -= 1
    if nh % 2 != 0: nh -= 1

    resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)

    r_c, g_c, b_c, a_c = resized.split()
    rgb_c = Image.merge('RGB', (r_c, g_c, b_c)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
    nr, ng, nb = rgb_c.split()
    sharpened = Image.merge('RGBA', (nr, ng, nb, a_c))

    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset_x = (target_w - nw) // 2
    offset_y = (target_h - nh) // 2
    canvas.paste(sharpened, (offset_x, offset_y), sharpened)
    return canvas

def build_scene_15_animation():
    raw_dir = "packs/pack_03_no_work/raw"
    out_dir = "packs/pack_03_no_work/output"
    os.makedirs(out_dir, exist_ok=True)

    pose_files = [
        os.path.join(raw_dir, "pose_04.png"), # 1. 呆滯放空看盤
        os.path.join(raw_dir, "pose_01.png"), # 2. 委屈抹淚抽泣
        os.path.join(raw_dir, "pose_03.png"), # 3. 仰天噴淚痛哭
        os.path.join(raw_dir, "pose_02.png"), # 4. 搖晃欄杆求救
    ]

    print("Preparing 4 clean keyframes...")
    key_images = [prepare_clean_keyframe(pf) for pf in pose_files]

    total_frames = 10
    frame_duration_ms = 250 # 10 * 250ms = 2500ms = 2.5s (or 200ms = 2.0s)
    target_w, target_h = 320, 270

    # Keyframe transition timeline (10 frames total)
    # Frame 0: Pose 4 (Hold 100%)
    # Frame 1: Pose 4 -> Pose 1 (Blend 50%)
    # Frame 2: Pose 1 (Hold 100%)
    # Frame 3: Pose 1 -> Pose 3 (Blend 40%)
    # Frame 4: Pose 3 (Hold 100% + shake)
    # Frame 5: Pose 3 (Hold 100% + violent shake)
    # Frame 6: Pose 3 -> Pose 2 (Blend 50%)
    # Frame 7: Pose 2 (Hold 100% + bar rattle)
    # Frame 8: Pose 2 (Hold 100% + bar rattle)
    # Frame 9: Pose 2 -> Pose 4 (Blend 50% loop back)

    timeline_plan = [
        (0, 0, 0.0, (0, 0)),      # F0: Pose 4 呆滯放空
        (0, 1, 0.6, (0, 0)),      # F1: 轉向抹淚
        (1, 1, 0.0, (0, 1)),      # F2: Pose 1 抹淚抽泣
        (1, 2, 0.5, (-1, -1)),    # F3: 轉向痛哭
        (2, 2, 0.0, (1, -2)),     # F4: Pose 3 仰天大噴淚 (震顫)
        (2, 2, 0.0, (-2, 2)),     # F5: Pose 3 仰天大噴淚 (震顫)
        (2, 3, 0.6, (0, 0)),      # F6: 撲上欄杆
        (3, 3, 0.0, (-3, 0)),     # F7: Pose 2 狂搖欄杆 LET ME OUT!
        (3, 3, 0.0, (3, 0)),      # F8: Pose 2 狂搖欄杆 LET ME OUT!
        (3, 0, 0.6, (0, 1)),      # F9: 無力滑落接回看盤
    ]

    all_frames = []

    for idx, (ka, kb, weight, (shk_x, shk_y)) in enumerate(timeline_plan):
        img_a = key_images[ka]
        img_b = key_images[kb]

        if weight == 0.0:
            frame_img = img_a
        else:
            frame_img = Image.blend(img_a, img_b, weight)

        canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        canvas.paste(frame_img, (shk_x, shk_y), frame_img)
        all_frames.append(canvas)

    # Export APNG (< 300KB)
    temp_files = []
    for idx, f in enumerate(all_frames):
        tmp_name = os.path.join(out_dir, f"temp_kf_{idx:02d}.png")
        qf = f.quantize(colors=72, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    out_apng = os.path.join(out_dir, "15.png")
    apng = APNG(num_plays=0)
    for tmp_name in temp_files:
        apng.append_file(tmp_name, delay=frame_duration_ms, delay_den=1000)

    apng.save(out_apng)

    # Export GIF preview
    out_gif = os.path.join(out_dir, "15_preview.gif")
    all_frames[0].save(
        out_gif,
        save_all=True,
        append_images=all_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2
    )

    for tmp_name in temp_files:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # Export 5x2 contact sheet
    fw, fh = all_frames[0].size
    sheet = Image.new('RGBA', (fw * 5, fh * 2), (240, 240, 240, 255))
    for idx, f in enumerate(all_frames):
        r = idx // 5
        c = idx % 5
        sheet.paste(f, (c * fw, r * fh), f)
    sheet_path = os.path.join(out_dir, "15_contact_sheet.png")
    sheet.save(sheet_path)

    print(f"\n==========================================")
    print(f"[SUCCESS] Generated LINE APNG: {out_apng} ({os.path.getsize(out_apng)/1024:.1f} KB)")
    print(f"[SUCCESS] Generated Preview GIF: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")
    print(f"[SUCCESS] Generated Contact Sheet: {sheet_path}")
    print(f"==========================================")

if __name__ == "__main__":
    build_scene_15_animation()
