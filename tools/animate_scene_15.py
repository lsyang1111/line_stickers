"""
animate_scene_15.py - 為「住套房流淚」製作精緻連貫角色動畫

動畫設計：
1. 淚水瀑布波動 (Flowing Tear Streams)：眼淚水流產生向下流動的漣漪波浪與淚滴落下效果
2. 抽泣發抖震顫 (Sobbing & Shivering)：雙貓身體與頭部隨悲傷哭泣產生細緻抽動與微震顫
3. 淒涼落葉飄零 (Swaying Falling Leaf)：右側落葉隨冷風左右搖曳、旋轉並向下緩慢飄落
4. 冷風吹拂流動 (Wind Swirl Breeze)：冷風線條從左至右飄移
5. 輸出符合 LINE 動態貼圖 APNG 標準 (10格, 1.0秒完美循環, 尺寸符合規範, 檔案 < 300KB)
"""

import os
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from apng import APNG

def build_animated_scene_15():
    raw_path = "packs/pack_02_no_work/raw/scene_15.png"
    out_dir = "packs/pack_02_no_work/animated_demo"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error loading {raw_path}")
        return

    h, w = img.shape[:2]
    b, g, r, a = cv2.split(img)

    # 1. 精準定位淚水區域 (Cyan/Blue Tears)
    # Tears bounding boxes:
    # Cat 1 (left): x: 330..490, y: 400..560
    # Cat 2 (right): x: 530..710, y: 400..560
    tear_mask = np.zeros((h, w), dtype=np.uint8)
    tear_cond = (b.astype(int) > 170) & (g.astype(int) > 130) & (r.astype(int) < 140) & (a > 50)
    tear_mask[400:580, 330:720] = (tear_cond[400:580, 330:720]).astype(np.uint8) * 255

    # 2. 精準定位落葉 (Green Leaf at top right)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    leaf_cond = (g.astype(int) > r.astype(int) + 20) & (g.astype(int) > b.astype(int) + 20) & (a > 100)
    leaf_mask[320:410, 950:1020] = (leaf_cond[320:410, 950:1020]).astype(np.uint8) * 255
    # Dilate slightly to capture complete leaf
    leaf_mask = cv2.dilate(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # 3. 精準定位冷風線 (Wind lines)
    wind_mask = np.zeros((h, w), dtype=np.uint8)
    wind_cond = (b.astype(int) > 180) & (g.astype(int) > 160) & (r.astype(int) > 120) & (r.astype(int) < 220) & (a > 30)
    wind_mask[380:580, 120:300] = (wind_cond[380:580, 120:300]).astype(np.uint8) * 255
    wind_mask[580:750, 720:900] = (wind_cond[580:750, 720:900]).astype(np.uint8) * 255

    # 分離圖層
    # 底層 (Base cats + bars + sign)
    base_img = img.copy()
    # Remove leaf from base by inpainting/masking
    leaf_y, leaf_x = np.where(leaf_mask > 0)
    if len(leaf_x) > 0:
        base_img[leaf_mask > 0] = [0, 0, 0, 0]
        leaf_patch = img[min(leaf_y):max(leaf_y)+1, min(leaf_x):max(leaf_x)+1].copy()
        leaf_center = ((min(leaf_x) + max(leaf_x)) // 2, (min(leaf_y) + max(leaf_y)) // 2)
    else:
        leaf_patch = None

    total_frames = 10
    frame_duration_ms = 100 # 10 * 100ms = 1000ms = 1.0s loop

    # 抽泣發抖震顫曲線 (Shivering Sobbing Offsets)
    shiver_y = [0, -3, 2, -2, 3, -1, 2, -3, 1, 0]
    shiver_x = [0, 1, -1, 1, 0, -1, 1, 0, -1, 0]

    frames_pil = []

    for i in range(total_frames):
        t = i / float(total_frames)
        frame_canvas = np.zeros_like(img)

        # A. 雙貓本體抽搐顫抖 (Cats Shivering)
        dy = shiver_y[i]
        dx = shiver_x[i]
        M_body = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_base = cv2.warpAffine(base_img, M_body, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

        # B. 淚水波動水流 (Animated Flowing Tears)
        # Apply vertical wave mesh warping on tears
        phase = t * 2 * math.pi
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        # Only displace within tear mask region
        tear_displacement_x = np.sin(map_y / 16.0 + phase) * 2.8 * (tear_mask / 255.0)
        tear_displacement_y = (np.cos(map_y / 18.0 + phase) * 2.0 + dy) * (tear_mask / 255.0)

        grid_x = (map_x + tear_displacement_x).astype(np.float32)
        grid_y = (map_y + tear_displacement_y).astype(np.float32)

        warped_tears = cv2.remap(img, grid_x, grid_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
        
        # 疊合淚水到本體
        # Replace tears with animated version
        tear_alpha = (tear_mask > 0)[:, :, None]
        composite = np.where(tear_alpha, warped_tears, shifted_base)

        # C. 淚滴下落與噴出 (Falling Teardrops)
        # Add comical dripping tear beads
        drop_t = (t * 2.0) % 1.0
        drop_y_offset = int(drop_t * 35)
        drop_alpha = math.sin(drop_t * math.pi)
        
        # Teardrop positions below eyes
        drop_points = [(360, 520 + drop_y_offset), (460, 520 + drop_y_offset), (565, 520 + drop_y_offset), (680, 520 + drop_y_offset)]
        for pt in drop_points:
            cv2.circle(composite, pt, int(4 * drop_alpha + 2), (240, 210, 130, int(220 * drop_alpha)), -1)
            cv2.circle(composite, pt, int(5 * drop_alpha + 2), (255, 255, 255, int(200 * drop_alpha)), 1)

        # D. 落葉飄零旋轉 (Swaying Falling Leaf)
        if leaf_patch is not None:
            leaf_h, leaf_w = leaf_patch.shape[:2]
            leaf_rot = math.sin(t * 2 * math.pi) * 25.0 # rotate -25 to +25 deg
            leaf_drift_x = math.sin(t * 2 * math.pi) * 18.0
            leaf_drift_y = math.sin(t * math.pi) * 35.0 # drift down and back in seamless loop

            # Rotate leaf patch
            M_leaf = cv2.getRotationMatrix2D((leaf_w / 2, leaf_h / 2), leaf_rot, 1.0)
            rot_leaf = cv2.warpAffine(leaf_patch, M_leaf, (leaf_w, leaf_h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

            # Paste leaf onto canvas
            target_lx = int(min(leaf_x) + leaf_drift_x)
            target_ly = int(min(leaf_y) + leaf_drift_y)

            # Blend leaf
            for ly in range(leaf_h):
                for lx in range(leaf_w):
                    cy = target_ly + ly
                    cx = target_lx + lx
                    if 0 <= cy < h and 0 <= cx < w:
                        la = rot_leaf[ly, lx, 3] / 255.0
                        if la > 0:
                            bg_a = composite[cy, cx, 3] / 255.0
                            out_a = la + bg_a * (1 - la)
                            if out_a > 0:
                                composite[cy, cx, :3] = (rot_leaf[ly, lx, :3] * la + composite[cy, cx, :3] * bg_a * (1 - la)) / out_a
                                composite[cy, cx, 3] = int(out_a * 255)

        # E. 冷風流動微移 (Wind Breeze Drifting)
        wind_shift_x = int(math.sin(t * 2 * math.pi) * 8.0)
        M_wind = np.float32([[1, 0, wind_shift_x], [0, 1, 0]])
        warped_wind = cv2.warpAffine(composite, M_wind, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
        wind_alpha_mask = (wind_mask > 0)[:, :, None]
        composite = np.where(wind_alpha_mask, warped_wind, composite)

        # F. 轉換為 PIL 並縮放至 LINE 動畫貼圖規範
        # LINE specs: max 320x270, one side must be exactly 270 (even numbers)
        pil_frame = Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGRA2RGBA))
        
        # Crop tight content
        alpha_np = composite[:, :, 3]
        y_idxs, x_idxs = np.where(alpha_np > 10)
        x0, x1 = int(np.min(x_idxs)), int(np.max(x_idxs))
        y0, y1 = int(np.min(y_idxs)), int(np.max(y_idxs))
        
        # Add 10px margin around content
        x0 = max(0, x0 - 10)
        y0 = max(0, y0 - 10)
        x1 = min(w - 1, x1 + 10)
        y1 = min(h - 1, y1 + 10)
        
        cropped_frame = pil_frame.crop((x0, y0, x1 + 1, y1 + 1))
        
        # Resize to fit inside 320x270 with height=270
        cw, ch = cropped_frame.size
        scale = min(320.0 / cw, 270.0 / ch)
        nw = int(round(cw * scale))
        nh = int(round(ch * scale))
        if nw % 2 != 0: nw -= 1
        if nh % 2 != 0: nh -= 1
        
        # Ensure at least one side is exactly 270
        canvas_w = min(320, max(nw, 270))
        canvas_h = 270
        if canvas_w % 2 != 0: canvas_w -= 1
        
        resized_frame = cropped_frame.resize((nw, nh), Image.Resampling.LANCZOS)
        
        # Unsharp mask for crisp HD fur and text
        r_c, g_c, b_c, a_c = resized_frame.split()
        rgb_c = Image.merge('RGB', (r_c, g_c, b_c)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        nr, ng, nb = rgb_c.split()
        sharpened_frame = Image.merge('RGBA', (nr, ng, nb, a_c))
        
        # Paste on centered canvas
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        cx_pos = (canvas_w - nw) // 2
        cy_pos = (canvas_h - nh) // 2
        canvas.paste(sharpened_frame, (cx_pos, cy_pos), sharpened_frame)
        
        frames_pil.append(canvas)

    # 4. 輸出 APNG 檔 (LINE 官方規範)
    temp_files = []
    for idx, f in enumerate(frames_pil):
        tmp_name = os.path.join(out_dir, f"temp_f_{idx:02d}.png")
        # Quantize to 128 colors to stay well under 300KB limit
        qf = f.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    apng = APNG(num_plays=0) # loop infinite for preview, num_plays=1 or 0
    for tmp_name in temp_files:
        apng.append_file(tmp_name, delay=frame_duration_ms, delay_den=1000)

    out_apng = os.path.join(out_dir, "scene_15.png")
    apng.save(out_apng)

    # 輸出 GIF 方便使用者在任何檢視器立即預覽動態效果
    out_gif = os.path.join(out_dir, "scene_15_preview.gif")
    frames_pil[0].save(
        out_gif,
        save_all=True,
        append_images=frames_pil[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2
    )

    # Clean up temp frames
    for tmp_name in temp_files:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    print(f"[SUCCESS] Generated Animated APNG: {out_apng} ({os.path.getsize(out_apng)/1024:.1f} KB)")
    print(f"[SUCCESS] Generated Preview GIF: {out_gif} ({os.path.getsize(out_gif)/1024:.1f} KB)")

if __name__ == "__main__":
    build_animated_scene_15()
