"""
animate_scene_01_physical.py - 不想上班（貓壽司）真實貓咪肢體與表情解剖級動畫

動作編排 (12 Frames / 3.0s Loop)：
1. 貓咪深呼吸起伏 (Chest Breathing Expansion)
2. 黑貓貓爪探出棉被抓抓 (Paw Extending & Flexing from Blanket)
3. 花貓張嘴打大呵欠露出粉紅舌頭 (Wide Yawn with Pink Tongue & Squinting Eyes)
4. 貓耳朵向後壓成飛機耳與抖動 (Airplane Ears Twitching & Drooping)
5. 緩慢眨眼 (Sleepy Eyelid Blink)
6. 縮回被窩蹭蹭 (Snuggle into Blanket Sanctuary)
7. 100% 符合 LINE 官方 APNG 規範 (< 300KB, 320x270, 偶數透明畫布)
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter
from apng import APNG

def remove_bg(img_bgra, tolerance=(25, 25, 25)):
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

def ensure_white_border(img_bgra, border_px=6):
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

def build_scene_01_physical_animation():
    raw_path = "packs/pack_03_no_work/raw/scene_01.png"
    out_dir = "packs/pack_03_no_work/output"
    os.makedirs(out_dir, exist_ok=True)

    img_raw = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot load {raw_path}")

    # Scale raw to 1195x896 for ultra-fast and precise deformation processing
    img = cv2.resize(img_raw, (1195, 896), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # Pre-clean background
    if img.shape[2] == 3 or not np.any(img[:, :, 3] < 250):
        img = remove_bg(img)

    # 1. 建立精準解剖特徵權重圖 (Landmark Region Masks)
    grid_y, grid_x = np.ogrid[:h, :w]

    # A. 黑貓耳朵 (Left: 260, 250; Right: 430, 250)
    d_bear_l = np.sqrt(((grid_x - 260)/60.0)**2 + ((grid_y - 240)/70.0)**2)
    mask_bear_l = np.clip(1.0 - d_bear_l, 0.0, 1.0)**2

    # B. 黑貓眼睛 (Left: 320, 360; Right: 410, 360)
    d_beye = np.minimum(
        np.sqrt(((grid_x - 320)/40.0)**2 + ((grid_y - 360)/30.0)**2),
        np.sqrt(((grid_x - 410)/40.0)**2 + ((grid_y - 360)/30.0)**2)
    )
    mask_beye = np.clip(1.0 - d_beye, 0.0, 1.0)**2

    # C. 花貓嘴巴呵欠區域 (Mouth at 725, 440)
    d_cmouth = np.sqrt(((grid_x - 725)/55.0)**2 + ((grid_y - 440)/45.0)**2)
    mask_cmouth = np.clip(1.0 - d_cmouth, 0.0, 1.0)**2

    # D. 花貓眼睛瞇眼 (Left: 665, 370; Right: 780, 370)
    d_ceye = np.minimum(
        np.sqrt(((grid_x - 665)/45.0)**2 + ((grid_y - 370)/30.0)**2),
        np.sqrt(((grid_x - 780)/45.0)**2 + ((grid_y - 370)/30.0)**2)
    )
    mask_ceye = np.clip(1.0 - d_ceye, 0.0, 1.0)**2

    # E. 花貓耳朵 (Left: 630, 250; Right: 825, 260)
    d_cear = np.minimum(
        np.sqrt(((grid_x - 630)/60.0)**2 + ((grid_y - 250)/70.0)**2),
        np.sqrt(((grid_x - 825)/60.0)**2 + ((grid_y - 260)/70.0)**2)
    )
    mask_cear = np.clip(1.0 - d_cear, 0.0, 1.0)**2

    # F. 棉被深呼吸 (Center at 540, 650)
    d_blanket = np.sqrt(((grid_x - 540)/320.0)**2 + ((grid_y - 650)/200.0)**2)
    mask_blanket = np.clip(1.0 - d_blanket, 0.0, 1.0)**1.5

    # 2. 建立毛茸茸黑貓爪子 (Cute Black Paw Asset)
    paw_w, paw_h = 75, 75
    paw_layer = np.zeros((paw_h, paw_w, 4), dtype=np.uint8)
    # Draw cute black cat paw with pink toe beans
    cv2.ellipse(paw_layer, (38, 42), (28, 22), 0, 0, 360, (25, 25, 28, 255), -1) # Main pad
    # 4 little toe beans
    cv2.circle(paw_layer, (18, 24), 9, (25, 25, 28, 255), -1)
    cv2.circle(paw_layer, (31, 16), 10, (25, 25, 28, 255), -1)
    cv2.circle(paw_layer, (46, 16), 10, (25, 25, 28, 255), -1)
    cv2.circle(paw_layer, (59, 24), 9, (25, 25, 28, 255), -1)
    # Pink pads
    cv2.ellipse(paw_layer, (38, 44), (16, 11), 0, 0, 360, (160, 140, 235, 255), -1)
    cv2.circle(paw_layer, (18, 25), 5, (160, 140, 235, 255), -1)
    cv2.circle(paw_layer, (31, 17), 5, (160, 140, 235, 255), -1)
    cv2.circle(paw_layer, (46, 17), 5, (160, 140, 235, 255), -1)
    cv2.circle(paw_layer, (59, 25), 5, (160, 140, 235, 255), -1)
    # White die-cut outline around paw
    paw_alpha = paw_layer[:, :, 3]
    paw_dilated = cv2.dilate(paw_alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    paw_white = np.full((paw_h, paw_w, 4), 255, dtype=np.uint8)
    paw_white[:, :, 3] = cv2.GaussianBlur(paw_dilated, (3, 3), 0)
    paw_combined = Image.alpha_composite(Image.fromarray(paw_white), Image.fromarray(paw_layer))
    paw_final_bgra = cv2.cvtColor(np.array(paw_combined), cv2.COLOR_RGBA2BGRA)

    total_frames = 12
    frame_duration_ms = 250 # 12 * 250ms = 3000ms = 3.0s loop

    map_x_base, map_y_base = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    all_frames_pil = []

    for i in range(total_frames):
        t = i / float(total_frames)
        sin_t = math.sin(t * 2 * math.pi)

        # -------------------------------------------------------------
        # 動作 1: 棉被規律深呼吸起伏 (Breathing Expansion)
        # -------------------------------------------------------------
        breath_scale = math.sin(t * 4 * math.pi) * 8.0 # 2 breath cycles per loop
        disp_breath_y = breath_scale * mask_blanket
        disp_breath_x = math.sin(t * 4 * math.pi) * 4.0 * mask_blanket

        # -------------------------------------------------------------
        # 動作 2: 黑貓耳朵抖動 (Ear Twitch at t=0.1~0.25)
        # -------------------------------------------------------------
        if 0.08 <= t <= 0.25:
            ear_t = (t - 0.08) / 0.17
            twitch_rot = math.sin(ear_t * 4 * math.pi) * 8.0
        else:
            twitch_rot = 0.0
        disp_bear_x = twitch_rot * mask_bear_l

        # -------------------------------------------------------------
        # 動作 3: 花貓打大呵欠 (Yawn Expansion at t=0.25~0.65)
        # -------------------------------------------------------------
        if 0.25 <= t <= 0.65:
            yawn_t = (t - 0.25) / 0.40 # 0.0 -> 1.0 -> 0.0
            yawn_intensity = math.sin(yawn_t * math.pi) # Peak at t=0.45
            
            # Mouth opens downwards
            disp_yawn_mouth_y = -yawn_intensity * 18.0 * mask_cmouth
            # Eyes squeeze shut
            disp_yawn_eye_y = yawn_intensity * 8.0 * mask_ceye
            # Airplane ears flatten outwards and downwards
            disp_cear_y = yawn_intensity * 12.0 * mask_cear
            disp_cear_x = (np.sign(grid_x - 725) * yawn_intensity * 8.0) * mask_cear
        else:
            yawn_intensity = 0.0
            disp_yawn_mouth_y = 0.0
            disp_yawn_eye_y = 0.0
            disp_cear_y = 0.0
            disp_cear_x = 0.0

        # -------------------------------------------------------------
        # 動作 4: 黑貓眨眼 (Sleepy Blink at t=0.70~0.85)
        # -------------------------------------------------------------
        if 0.70 <= t <= 0.85:
            blink_t = (t - 0.70) / 0.15
            blink_intensity = math.sin(blink_t * math.pi)
            disp_beye_y = blink_intensity * 10.0 * mask_beye
        else:
            blink_intensity = 0.0
            disp_beye_y = 0.0

        # -------------------------------------------------------------
        # 總位移場映射 (Mesh Deformation)
        # -------------------------------------------------------------
        total_dx = disp_breath_x + disp_bear_x + disp_cear_x
        total_dy = disp_breath_y + disp_yawn_mouth_y + disp_yawn_eye_y + disp_cear_y + disp_beye_y

        sample_x = (map_x_base - total_dx).astype(np.float32)
        sample_y = (map_y_base - total_dy).astype(np.float32)

        warped = cv2.remap(img, sample_x, sample_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

        # -------------------------------------------------------------
        # 動作 5: 花貓呵欠粉紅舌頭與小黑洞 (Yawn Mouth Cavity Overlay)
        # -------------------------------------------------------------
        if yawn_intensity > 0.3:
            y_rad = int(yawn_intensity * 16)
            x_rad = int(yawn_intensity * 18)
            mouth_center = (725, int(435 + disp_breath_y[435, 725]))
            # Dark open mouth cavity
            cv2.ellipse(warped, mouth_center, (x_rad, y_rad), 0, 0, 360, (20, 15, 45, int(240 * yawn_intensity)), -1)
            # Cute pink tongue inside
            tongue_center = (mouth_center[0], mouth_center[1] + int(y_rad * 0.45))
            cv2.ellipse(warped, tongue_center, (int(x_rad * 0.75), int(y_rad * 0.55)), 0, 0, 360, (140, 120, 245, int(250 * yawn_intensity)), -1)

        # -------------------------------------------------------------
        # 動作 6: 黑貓伸出貓爪抓抓 (Paw Extending from Blanket at t=0.30~0.75)
        # -------------------------------------------------------------
        if 0.30 <= t <= 0.75:
            paw_t = (t - 0.30) / 0.45
            paw_reach = math.sin(paw_t * math.pi) # 0.0 -> 1.0 -> 0.0
            paw_x_pos = int(500 - paw_reach * 45 + math.sin(paw_t * 4 * math.pi) * 6)
            paw_y_pos = int(480 - paw_reach * 70)
            paw_angle = math.sin(paw_t * 3 * math.pi) * 15.0 # waving paw

            M_paw = cv2.getRotationMatrix2D((paw_w/2.0, paw_h/2.0), paw_angle, 1.0)
            rot_paw = cv2.warpAffine(paw_final_bgra, M_paw, (paw_w, paw_h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

            # Alpha blend paw onto warped canvas
            for py in range(paw_h):
                for px in range(paw_w):
                    cy_p = paw_y_pos + py
                    cx_p = paw_x_pos + px
                    if 0 <= cy_p < h and 0 <= cx_p < w:
                        pa = (rot_paw[py, px, 3] / 255.0) * paw_reach
                        if pa > 0:
                            ba = warped[cy_p, cx_p, 3] / 255.0
                            oa = pa + ba * (1.0 - pa)
                            if oa > 0:
                                warped[cy_p, cx_p, :3] = (rot_paw[py, px, :3] * pa + warped[cy_p, cx_p, :3] * ba * (1.0 - pa)) / oa
                                warped[cy_p, cx_p, 3] = int(oa * 255)

        # -------------------------------------------------------------
        # 動作 7: 浮動 Zzz 睡眠泡泡 (Floating Zzz)
        # -------------------------------------------------------------
        for z_i in range(3):
            zt = (t + z_i / 3.0) % 1.0
            zx = int(880 + math.sin(zt * 2 * math.pi) * 15)
            zy = int(320 - zt * 70)
            z_alpha = math.sin(zt * math.pi)
            if 0 <= zy < h and 0 <= zx < w and z_alpha > 0.1:
                cv2.putText(warped, "z", (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.9 + zt*0.4, (200, 200, 255, int(220 * z_alpha)), 3)
                cv2.putText(warped, "z", (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.9 + zt*0.4, (80, 80, 160, int(220 * z_alpha)), 1)

        # -------------------------------------------------------------
        # 8. 縮放並封裝至 LINE 規格 (320x270 Canvas)
        # -------------------------------------------------------------
        pil_frame = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGRA2RGBA))
        
        alpha_np = warped[:, :, 3]
        y_idxs, x_idxs = np.where(alpha_np > 10)
        x0, x1 = int(np.min(x_idxs)), int(np.max(x_idxs))
        y0, y1 = int(np.min(y_idxs)), int(np.max(y_idxs))
        
        x0 = max(0, x0 - 8)
        y0 = max(0, y0 - 8)
        x1 = min(w - 1, x1 + 8)
        y1 = min(h - 1, y1 + 8)
        
        cropped_frame = pil_frame.crop((x0, y0, x1 + 1, y1 + 1))
        
        cw, ch = cropped_frame.size
        scale = min(320.0 / cw, 270.0 / ch)
        nw = int(round(cw * scale))
        nh = int(round(ch * scale))
        if nw % 2 != 0: nw -= 1
        if nh % 2 != 0: nh -= 1
        
        canvas_w = min(320, max(nw, 270))
        canvas_h = 270
        if canvas_w % 2 != 0: canvas_w -= 1
        
        resized_frame = cropped_frame.resize((nw, nh), Image.Resampling.LANCZOS)
        
        # HD 銳化
        r_c, g_c, b_c, a_c = resized_frame.split()
        rgb_c = Image.merge('RGB', (r_c, g_c, b_c)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=2))
        nr, ng, nb = rgb_c.split()
        sharpened_frame = Image.merge('RGBA', (nr, ng, nb, a_c))
        
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        cx_pos = (canvas_w - nw) // 2
        cy_pos = (canvas_h - nh) // 2
        canvas.paste(sharpened_frame, (cx_pos, cy_pos), sharpened_frame)
        
        all_frames_pil.append(canvas)

    # 輸出 APNG 與 GIF
    temp_files = []
    for idx, f in enumerate(all_frames_pil):
        tmp_name = os.path.join(out_dir, f"temp_p01_{idx:02d}.png")
        qf = f.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    out_apng = os.path.join(out_dir, "01.png")
    apng = APNG(num_plays=0)
    for tmp in temp_files:
        apng.append_file(tmp, delay=frame_duration_ms, delay_den=1000)

    apng.save(out_apng)

    out_gif = os.path.join(out_dir, "01_preview.gif")
    all_frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=all_frames_pil[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2
    )

    for tmp in temp_files:
        if os.path.exists(tmp):
            os.remove(tmp)

    # 更新 4x3 contact sheet
    fw, fh = all_frames_pil[0].size
    sheet = Image.new('RGBA', (fw * 4, fh * 3), (240, 240, 240, 255))
    for idx, f in enumerate(all_frames_pil):
        r = idx // 4
        c = idx % 4
        sheet.paste(f, (c * fw, r * fh), f)
    sheet_path = os.path.join(out_dir, "01_physical_sheet.png")
    sheet.save(sheet_path)

    print(f"\n==========================================")
    print(f"[SUCCESS PHYSICAL] Generated Anatomical APNG: {out_apng} ({os.path.getsize(out_apng)/1024:.1f} KB)")
    print(f"[SUCCESS PHYSICAL] Generated Preview GIF: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")
    print(f"[SUCCESS PHYSICAL] Generated Contact Sheet: {sheet_path}")
    print(f"==========================================")

if __name__ == "__main__":
    build_scene_01_physical_animation()
