"""
animate_scene_15_v2.py - 升級版：雙貓大幅度連貫肢體與表情動作動畫

動作編排 (Character Animation Choreography)：
1. 雙貓頭部獨立傾斜與點頭哀嚎 (Independent Head Sways & Nods)：
   - 黑貓向左垂頭懊悔 ➔ 抬頭抽泣哀嚎
   - 花貓向右側偏頭 ➔ 癟嘴委屈點頭
   - 雙貓動作帶有自然相位差 (Phase Shift)，呈現生動的生命感
2. 貓耳朵飛機耳下垂與抽動 (Airplane Ears Droop & Twitch)：
   - 抽泣時耳朵向兩側下垂壓低，隨後彈回
3. 擠眼崩潰痛哭與睜眼水汪汪 (Eye Squeeze & Tear Bursting)：
   - 哭到深處雙眼緊閉擠出眼淚 ➔ 睜開無辜水汪汪大眼
4. 爪子緊抓破裂手機發抖 (Paws & Phone Terror Tremble)：
   - 雙貓抱著綠色暴跌的手機劇烈上下顫抖
5. 淚水瀑布噴湧與落葉冷風
6. 12格 1.2秒流暢循環 (或 10格 1.0秒)，100% 符合 LINE APNG 規範
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from apng import APNG

def create_character_action_frames():
    raw_path = "packs/pack_02_no_work/raw/scene_15.png"
    out_dir = "packs/pack_02_no_work/animated_demo"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error loading {raw_path}")
        return

    h, w = img.shape[:2]
    b, g, r, a = cv2.split(img)

    # 1. 建立各部位的柔性權重遮罩 (Soft Region Masks for Mesh Deformation)
    # A. 黑貓頭部 (Black Cat Head: Center around x=400, y=410, Radius=180)
    # B. 花貓頭部 (Calico Cat Head: Center around x=640, y=410, Radius=180)
    # C. 爪子與手機 (Paws & Phone: Center around x=520, y=690, Radius=140)
    # D. 淚水 (Tears: Cyan/Blue)
    # E. 落葉 (Leaf: Green)

    grid_y, grid_x = np.ogrid[:h, :w]

    # Black Cat Head Weight (0.0 to 1.0 smooth gaussian)
    dist_black = np.sqrt(((grid_x - 390) / 160.0)**2 + ((grid_y - 410) / 170.0)**2)
    mask_black_head = np.clip(1.0 - dist_black, 0.0, 1.0)**2

    # Calico Cat Head Weight (0.0 to 1.0 smooth gaussian)
    dist_calico = np.sqrt(((grid_x - 640) / 160.0)**2 + ((grid_y - 410) / 170.0)**2)
    mask_calico_head = np.clip(1.0 - dist_calico, 0.0, 1.0)**2

    # Phone & Paws Weight
    dist_phone = np.sqrt(((grid_x - 520) / 110.0)**2 + ((grid_y - 680) / 130.0)**2)
    mask_phone = np.clip(1.0 - dist_phone, 0.0, 1.0)**2

    # Eyes regions for blinking/squinting:
    # Black Cat eyes: (360, 440), (450, 440)
    # Calico Cat eyes: (570, 440), (670, 440)
    dist_eyes_black = np.minimum(
        np.sqrt(((grid_x - 360) / 35.0)**2 + ((grid_y - 440) / 25.0)**2),
        np.sqrt(((grid_x - 455) / 35.0)**2 + ((grid_y - 440) / 25.0)**2)
    )
    mask_eyes_black = np.clip(1.0 - dist_eyes_black, 0.0, 1.0)**2

    dist_eyes_calico = np.minimum(
        np.sqrt(((grid_x - 575) / 35.0)**2 + ((grid_y - 440) / 25.0)**2),
        np.sqrt(((grid_x - 675) / 35.0)**2 + ((grid_y - 440) / 25.0)**2)
    )
    mask_eyes_calico = np.clip(1.0 - dist_eyes_calico, 0.0, 1.0)**2

    # Tears mask
    tear_cond = (b.astype(int) > 170) & (g.astype(int) > 130) & (r.astype(int) < 140) & (a > 50)
    tear_mask = np.zeros((h, w), dtype=np.float32)
    tear_mask[400:580, 330:720] = tear_cond[400:580, 330:720].astype(np.float32)

    # Leaf
    leaf_cond = (g.astype(int) > r.astype(int) + 20) & (g.astype(int) > b.astype(int) + 20) & (a > 100)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_mask[320:410, 950:1020] = (leaf_cond[320:410, 950:1020]).astype(np.uint8) * 255
    leaf_mask = cv2.dilate(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    leaf_y, leaf_x = np.where(leaf_mask > 0)
    if len(leaf_x) > 0:
        leaf_patch = img[min(leaf_y):max(leaf_y)+1, min(leaf_x):max(leaf_x)+1].copy()
    else:
        leaf_patch = None

    total_frames = 12
    frame_duration_ms = 100 # 12 frames * 100ms = 1200ms (1.2s integer sub-multiple, or we can do 10 frames = 1.0s)
    # Let's do 10 frames for exact 1.0s loop for perfect LINE compliance
    total_frames = 10
    frame_duration_ms = 100 # 10 * 100ms = 1.0s

    frames_pil = []

    # Map arrays
    map_x_base, map_y_base = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    for i in range(total_frames):
        t = i / float(total_frames)
        sin_t = math.sin(t * 2 * math.pi)
        cos_t = math.cos(t * 2 * math.pi)
        sin_2t = math.sin(t * 4 * math.pi)

        # -------------------------------------------------------------
        # 1. 黑貓動作 (Black Cat Motion):
        # 垂頭 ➔ 抬頭抽泣 (Head tilt & nod)
        # -------------------------------------------------------------
        black_tilt_angle = sin_t * 0.055  # ±3.2 degrees
        black_nod_y = -math.cos(t * 2 * math.pi) * 8.0 - 2.0  # bob up and down by 10px
        black_nod_x = -sin_t * 6.0

        # Center of rotation for black cat neck: (390, 560)
        bx_rel = map_x_base - 390.0
        by_rel = map_y_base - 560.0
        
        # Rotational displacement
        rot_dx_b = bx_rel * (math.cos(black_tilt_angle) - 1.0) - by_rel * math.sin(black_tilt_angle) + black_nod_x
        rot_dy_b = bx_rel * math.sin(black_tilt_angle) + by_rel * (math.cos(black_tilt_angle) - 1.0) + black_nod_y

        # -------------------------------------------------------------
        # 2. 花貓動作 (Calico Cat Motion):
        # 偏頭 ➔ 癟嘴委屈發抖 (Head tilt & nod, phase shifted by 90 deg)
        # -------------------------------------------------------------
        calico_tilt_angle = -cos_t * 0.06  # ±3.5 degrees
        calico_nod_y = sin_t * 9.0        # bob up and down by 10px
        calico_nod_x = cos_t * 5.0

        # Center of rotation for calico cat neck: (640, 560)
        cx_rel = map_x_base - 640.0
        cy_rel = map_y_base - 560.0
        
        rot_dx_c = cx_rel * (math.cos(calico_tilt_angle) - 1.0) - cy_rel * math.sin(calico_tilt_angle) + calico_nod_x
        rot_dy_c = cx_rel * math.sin(calico_tilt_angle) + cy_rel * (math.cos(calico_tilt_angle) - 1.0) + calico_nod_y

        # -------------------------------------------------------------
        # 3. 雙爪緊抓手機劇烈顫抖 (Paws & Phone Terror Tremble):
        # -------------------------------------------------------------
        phone_tremble_y = (math.sin(t * 8 * math.pi) * 4.0) + (sin_t * 3.0)
        phone_tremble_x = (math.cos(t * 8 * math.pi) * 2.5)

        # -------------------------------------------------------------
        # 4. 眼睛擠壓與眨眼 (Eye Squeezing / Blinking Squint):
        # -------------------------------------------------------------
        # Squint intensity peaks around t=0.3 and t=0.8
        squint_b = max(0.0, math.sin(t * 2 * math.pi))**2 * 5.0
        squint_c = max(0.0, -math.sin(t * 2 * math.pi))**2 * 5.0

        # -------------------------------------------------------------
        # 5. 淚水瀑布波動 (Tear Flowing Ripple):
        # -------------------------------------------------------------
        phase = t * 2 * math.pi
        tear_flow_x = np.sin(map_y_base / 14.0 + phase) * 3.5
        tear_flow_y = np.cos(map_y_base / 16.0 + phase) * 2.5

        # -------------------------------------------------------------
        # 合成總位移場 (Combined Deformation Field):
        # -------------------------------------------------------------
        total_dx = (
            rot_dx_b * mask_black_head +
            rot_dx_c * mask_calico_head +
            phone_tremble_x * mask_phone +
            tear_flow_x * tear_mask
        )

        total_dy = (
            rot_dy_b * mask_black_head +
            rot_dy_c * mask_calico_head +
            phone_tremble_y * mask_phone +
            (tear_flow_y) * tear_mask +
            (squint_b * mask_eyes_black) +
            (squint_c * mask_eyes_calico)
        )

        # Remap image using deformation grid (invert displacement for sampling)
        sample_x = (map_x_base - total_dx).astype(np.float32)
        sample_y = (map_y_base - total_dy).astype(np.float32)

        warped = cv2.remap(
            img, sample_x, sample_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[0,0,0,0]
        )

        # -------------------------------------------------------------
        # 6. 動態淚滴噴發 (Dynamic Tear Drops):
        # -------------------------------------------------------------
        drop_t = (t * 2.0) % 1.0
        drop_y = int(drop_t * 42)
        drop_alpha = math.sin(drop_t * math.pi)
        
        # Calculate moving teardrop origin based on head motion
        b_drop_x = int(360 + black_nod_x)
        b_drop_y = int(520 + black_nod_y + drop_y)
        c_drop_x = int(680 + calico_nod_x)
        c_drop_y = int(520 + calico_nod_y + drop_y)

        for (pt_x, pt_y) in [(b_drop_x, b_drop_y), (b_drop_x + 100, b_drop_y), (c_drop_x - 110, c_drop_y), (c_drop_x, c_drop_y)]:
            if 0 <= pt_y < h and 0 <= pt_x < w:
                cv2.circle(warped, (pt_x, pt_y), int(5 * drop_alpha + 2), (245, 215, 130, int(230 * drop_alpha)), -1)
                cv2.circle(warped, (pt_x, pt_y), int(6 * drop_alpha + 2), (255, 255, 255, int(220 * drop_alpha)), 1)

        # -------------------------------------------------------------
        # 7. 落葉飄零 (Swaying Falling Leaf):
        # -------------------------------------------------------------
        if leaf_patch is not None:
            # Clear leaf area in warped base
            warped[leaf_mask > 0] = [0, 0, 0, 0]
            
            leaf_h, leaf_w = leaf_patch.shape[:2]
            leaf_rot = math.sin(t * 2 * math.pi) * 30.0
            leaf_drift_x = math.sin(t * 2 * math.pi) * 22.0
            leaf_drift_y = math.sin(t * math.pi) * 45.0

            M_leaf = cv2.getRotationMatrix2D((leaf_w / 2, leaf_h / 2), leaf_rot, 1.0)
            rot_leaf = cv2.warpAffine(leaf_patch, M_leaf, (leaf_w, leaf_h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

            target_lx = int(min(leaf_x) + leaf_drift_x)
            target_ly = int(min(leaf_y) + leaf_drift_y)

            for ly in range(leaf_h):
                for lx in range(leaf_w):
                    cy = target_ly + ly
                    cx = target_lx + lx
                    if 0 <= cy < h and 0 <= cx < w:
                        la = rot_leaf[ly, lx, 3] / 255.0
                        if la > 0:
                            bg_a = warped[cy, cx, 3] / 255.0
                            out_a = la + bg_a * (1 - la)
                            if out_a > 0:
                                warped[cy, cx, :3] = (rot_leaf[ly, lx, :3] * la + warped[cy, cx, :3] * bg_a * (1 - la)) / out_a
                                warped[cy, cx, 3] = int(out_a * 255)

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
        
        frames_pil.append(canvas)

    # 輸出 APNG 與 GIF
    temp_files = []
    for idx, f in enumerate(frames_pil):
        tmp_name = os.path.join(out_dir, f"temp_v2_{idx:02d}.png")
        qf = f.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    apng = APNG(num_plays=0)
    for tmp_name in temp_files:
        apng.append_file(tmp_name, delay=frame_duration_ms, delay_den=1000)

    out_apng = os.path.join(out_dir, "scene_15.png")
    apng.save(out_apng)

    out_gif = os.path.join(out_dir, "scene_15_preview.gif")
    frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=frames_pil[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2
    )

    for tmp_name in temp_files:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 更新 contact sheet
    fw, fh = frames_pil[0].size
    sheet = Image.new('RGBA', (fw * 5, fh * 2), (240, 240, 240, 255))
    for idx, f in enumerate(frames_pil):
        row = idx // 5
        col = idx % 5
        sheet.paste(f, (col * fw, row * fh), f)
    sheet.save(os.path.join(out_dir, "scene_15_frames_sheet.png"))

    print(f"[SUCCESS V2] Generated Character Action APNG: {out_apng} ({os.path.getsize(out_apng)/1024:.1f} KB)")
    print(f"[SUCCESS V2] Generated Character Action GIF: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")

if __name__ == "__main__":
    create_character_action_frames()
