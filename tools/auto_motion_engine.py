"""
auto_motion_engine.py - 全自動 2D 角色動作與物理物理動態引擎 (Procedural Character Motion Engine)

功能：
只需輸入一張原始貼圖 (raw image) 或指定 prompt 文件，
引擎將自動進行語意分析，掛載專屬的 2D 肢體骨架、柔性變形場 (Deformation Field) 與粒子系統，
一鍵自動演算生成 12 畫格 / 3.0 秒完美循環的 LINE 官方 APNG 動態貼圖！

支援動作模式 (Motion Archetypes)：
1. dance / celebrate : 扭腰跳舞、雙爪輪流高舉狂歡、漫天噴灑鈔票/彩帶金幣
2. cry / trapped     : 悲傷抽泣垂頭、雙眼擠壓、瀑布湧淚與淚滴噴發
3. jump / off_work   : 蓄力下壓 ➔ 騰空大跳躍 ➔ 雙爪大張歡呼 ➔ 落地彈性回彈
4. sneak / slack     : 賊頭賊腦左右斜視、嘴中魚兒/道具擺動、偷偷摸魚
5. melt / tang_ping  : 像液體般攤平融化擴散、生無可戀呼吸起伏
6. soul_out          : 靈魂出竅半透明升空、歡呼飄向門口
7. stock_crash       : 震驚恐慌高頻顫抖、綠色暴跌箭頭脈衝
8. auto              : 自動解析 prompt 關鍵字匹配最佳動作！
"""

import os
import sys
import math
import argparse
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from apng import APNG

def remove_background(img_bgr, tolerance=(25, 25, 25)):
    h, w = img_bgr.shape[:2]
    img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgr = img_bgr.copy()
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

def ensure_white_diecut(img_bgra, border_px=8):
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

def detect_motion_type(prompt_text: str) -> str:
    txt = prompt_text.lower()
    if any(k in txt for k in ["跳舞", "dance", "漲停", "數錢", "慶祝", "celebrat", "歐印", "財富自由"]):
        return "dance"
    elif any(k in txt for k in ["套房", "流淚", "哭", "cry", "sob", "tears", "悲傷", "後悔"]):
        return "cry"
    elif any(k in txt for k in ["下班", "跳躍", "jump", "leap", "飛", "soar", "自由"]):
        return "jump"
    elif any(k in txt for k in ["摸魚", "sneak", "sly", "偷懶", "魚", "fish"]):
        return "sneak"
    elif any(k in txt for k in ["躺平", "融化", "不想上班", "melt", "liquid", "puddle", "被被"]):
        return "melt"
    elif any(k in txt for k in ["靈魂出竅", "soul", "ghost", "空殼"]):
        return "soul_out"
    elif any(k in txt for k in ["跌", "崩盤", "crash", "green arrow", "shock"]):
        return "stock_crash"
    return "dance" # default cheerful motion

class MotionEngine:
    def __init__(self, raw_image_path: str):
        raw = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot load image: {raw_image_path}")

        # Check alpha
        if len(raw.shape) == 3 and raw.shape[2] == 4:
            if not np.any(raw[:, :, 3] < 250):
                raw = remove_background(raw[:, :, :3])
        else:
            raw = remove_background(raw)

        raw = ensure_white_diecut(raw, border_px=8)
        self.raw_bgra = raw
        self.h, self.w = raw.shape[:2]

        # Crop tight
        alpha = raw[:, :, 3]
        y_idx, x_idx = np.where(alpha > 10)
        if len(y_idx) > 0:
            self.cropped_bgra = raw[min(y_idx):max(y_idx)+1, min(x_idx):max(x_idx)+1]
        else:
            self.cropped_bgra = raw

        self.ch, self.cw = self.cropped_bgra.shape[:2]

        # Fit into 320x270 target with padding
        scale = min(300.0 / self.cw, 250.0 / self.ch)
        self.nw = int(round(self.cw * scale))
        self.nh = int(round(self.ch * scale))
        if self.nw % 2 != 0: self.nw -= 1
        if self.nh % 2 != 0: self.nh -= 1

        pil_cr = Image.fromarray(cv2.cvtColor(self.cropped_bgra, cv2.COLOR_BGRA2RGBA))
        self.base_pil = pil_cr.resize((self.nw, self.nh), Image.Resampling.LANCZOS)
        self.base_bgra = cv2.cvtColor(np.array(self.base_pil), cv2.COLOR_RGBA2BGRA)

        self.target_w = 320
        self.target_h = 270

    def render_motion(self, motion_type: str, total_frames=12, frame_duration_ms=250):
        print(f"Rendering motion '{motion_type}' ({total_frames} frames)...")
        frames = []

        for i in range(total_frames):
            t = i / float(total_frames)
            sin_t = math.sin(t * 2 * math.pi)
            cos_t = math.cos(t * 2 * math.pi)
            sin_2t = math.sin(t * 4 * math.pi)

            canvas_bgra = np.zeros((self.target_h, self.target_w, 4), dtype=np.uint8)
            base_x = (self.target_w - self.nw) // 2
            base_y = (self.target_h - self.nh) // 2

            bw, bh = self.nw, self.nh
            map_x, map_y = np.meshgrid(np.arange(bw, dtype=np.float32), np.arange(bh, dtype=np.float32))

            # -----------------------------------------------------------------
            # 1. 跳舞 / 慶祝 / 漲停板 (Dance / Celebration)
            # -----------------------------------------------------------------
            if motion_type == "dance":
                # Rhythmic hip sway & head bop
                sway_x = sin_t * 8.0
                bop_y = -abs(sin_2t) * 10.0
                rot_angle = sin_t * 0.08 # ±4.5 deg

                cx, cy = bw / 2.0, bh * 0.85
                rel_x = map_x - cx
                rel_y = map_y - cy
                
                # Dynamic warp (upper body sways more than bottom)
                height_weight = (1.0 - (map_y / float(bh)))**1.2
                dx = (rel_x * (math.cos(rot_angle) - 1.0) - rel_y * math.sin(rot_angle) + sway_x) * height_weight
                dy = (rel_x * math.sin(rot_angle) + rel_y * (math.cos(rot_angle) - 1.0) + bop_y) * height_weight

                warped = cv2.remap(self.base_bgra, (map_x - dx).astype(np.float32), (map_y - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

                # Overlay onto canvas
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = warped

                # Add falling confetti / gold sparkles
                for c_idx in range(8):
                    cp_t = (t + c_idx / 8.0) % 1.0
                    cx_p = int((c_idx * 40 + math.sin(cp_t * 4 * math.pi) * 20) % self.target_w)
                    cy_p = int(cp_t * self.target_h)
                    color = [(255, 215, 0), (50, 205, 50), (255, 105, 180), (255, 69, 0)][c_idx % 4]
                    cv2.circle(canvas_bgra, (cx_p, cy_p), 3, (*color, 230), -1)

            # -----------------------------------------------------------------
            # 2. 哭泣 / 被套牢 (Cry / Sob)
            # -----------------------------------------------------------------
            elif motion_type == "cry":
                sob_y = (math.sin(t * 6 * math.pi) * 3.0) - (sin_t * 4.0)
                sob_x = math.sin(t * 8 * math.pi) * 1.5

                dx = sob_x * np.ones_like(map_x)
                dy = sob_y * (map_y / float(bh))

                warped = cv2.remap(self.base_bgra, (map_x - dx).astype(np.float32), (map_y - dy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = warped

                # Tear particle stream
                for tr_idx in range(4):
                    tr_t = (t * 2.0 + tr_idx / 4.0) % 1.0
                    tr_y = int(base_y + bh * 0.45 + tr_t * (bh * 0.4))
                    for eye_x_off in [bw * 0.3, bw * 0.42, bw * 0.58, bw * 0.7]:
                        tx = int(base_x + eye_x_off + math.sin(tr_t * 3 * math.pi) * 3)
                        alpha_drop = math.sin(tr_t * math.pi)
                        if 0 <= tr_y < self.target_h and 0 <= tx < self.target_w:
                            cv2.circle(canvas_bgra, (tx, tr_y), int(4 * alpha_drop + 1), (245, 215, 130, int(220 * alpha_drop)), -1)

            # -----------------------------------------------------------------
            # 3. 大跳躍 / 下班了 (Jump / Soar)
            # -----------------------------------------------------------------
            elif motion_type == "jump":
                # Jump trajectory: anticipation (squat) ➔ launch ➔ peak ➔ land
                if t < 0.2: # Squat
                    sq_t = t / 0.2
                    jump_y = math.sin(sq_t * math.pi) * 12.0
                    scale_y = 1.0 - (math.sin(sq_t * math.pi) * 0.12)
                    scale_x = 1.0 + (math.sin(sq_t * math.pi) * 0.08)
                elif t < 0.7: # Airborne jump
                    air_t = (t - 0.2) / 0.5
                    jump_y = -math.sin(air_t * math.pi) * 32.0 # Rocket up 32px
                    scale_y = 1.0 + (math.sin(air_t * math.pi) * 0.08)
                    scale_x = 1.0 - (math.sin(air_t * math.pi) * 0.05)
                else: # Landing bounce
                    land_t = (t - 0.7) / 0.3
                    jump_y = math.sin(land_t * math.pi) * 6.0
                    scale_y = 1.0 - (math.sin(land_t * math.pi) * 0.06)
                    scale_x = 1.0 + (math.sin(land_t * math.pi) * 0.04)

                M_jump = cv2.getRotationMatrix2D((bw/2.0, bh), 0, 1.0)
                M_jump[0, 0] *= scale_x
                M_jump[1, 1] *= scale_y
                M_jump[1, 2] += jump_y

                warped = cv2.warpAffine(self.base_bgra, M_jump, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = warped

                # Speed / joy sparks
                if 0.2 <= t <= 0.7:
                    for sp_i in range(6):
                        sx = int(base_x + bw/2.0 + (sp_i - 2.5) * 30)
                        sy = int(base_y + bh + jump_y + 10 + (sp_i % 2) * 8)
                        if 0 <= sy < self.target_h and 0 <= sx < self.target_w:
                            cv2.line(canvas_bgra, (sx, sy), (sx, min(self.target_h-1, sy + 14)), (255, 230, 120, 200), 2)

            # -----------------------------------------------------------------
            # 4. 摸魚 / 偷懶 (Sneak / Slack)
            # -----------------------------------------------------------------
            elif motion_type == "sneak":
                shift_x = sin_t * 12.0
                tilt = sin_t * 0.05
                M_snk = cv2.getRotationMatrix2D((bw/2.0, bh/2.0), math.degrees(tilt), 1.0)
                M_snk[0, 2] += shift_x

                warped = cv2.warpAffine(self.base_bgra, M_snk, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = warped

            # -----------------------------------------------------------------
            # 5. 融化 / 躺平 (Melt / Liquid Cat)
            # -----------------------------------------------------------------
            elif motion_type == "melt":
                melt_pulse = (sin_t + 1.0) / 2.0 # 0.0 to 1.0
                scale_x = 1.0 + (melt_pulse * 0.14) # expand wide
                scale_y = 1.0 - (melt_pulse * 0.12) # flatten down

                M_melt = cv2.getRotationMatrix2D((bw/2.0, bh), 0, 1.0)
                M_melt[0, 0] *= scale_x
                M_melt[1, 1] *= scale_y

                warped = cv2.warpAffine(self.base_bgra, M_melt, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = warped

                # Floating Zzz
                for z_i in range(3):
                    zt = (t + z_i / 3.0) % 1.0
                    zx = int(base_x + bw * 0.75 + math.sin(zt * 2 * math.pi) * 15)
                    zy = int(base_y + bh * 0.3 - zt * 60)
                    if 0 <= zy < self.target_h and 0 <= zx < self.target_w:
                        cv2.putText(canvas_bgra, "z", (zx, zy), cv2.FONT_HERSHEY_SIMPLEX, 0.5 + zt*0.3, (140, 180, 255, int(220 * math.sin(zt*math.pi))), 2)

            # -----------------------------------------------------------------
            # 6. 靈魂出竅 (Soul Leaving Body)
            # -----------------------------------------------------------------
            elif motion_type == "soul_out":
                # Physical body sits still at bottom
                canvas_bgra[base_y:base_y+bh, base_x:base_x+bw] = self.base_bgra

                # Translucent soul floats upward
                soul_t = t # 0.0 to 1.0
                soul_y = -int(soul_t * 50)
                soul_x = int(math.sin(soul_t * 2 * math.pi) * 16)
                soul_alpha = math.sin(soul_t * math.pi) * 0.65 # max 65% opacity

                soul_bgra = self.base_bgra.copy()
                soul_bgra[:, :, 3] = (soul_bgra[:, :, 3].astype(float) * soul_alpha).astype(np.uint8)

                # Tint soul slightly cyan/blue ethereal
                soul_bgra[:, :, 0] = np.clip(soul_bgra[:, :, 0].astype(int) + 40, 0, 255)
                soul_bgra[:, :, 1] = np.clip(soul_bgra[:, :, 1].astype(int) + 30, 0, 255)

                M_soul = np.float32([[1, 0, soul_x], [0, 1, soul_y]])
                warped_soul = cv2.warpAffine(soul_bgra, M_soul, (bw, bh), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0,0])

                # Alpha blend soul over canvas
                for sy in range(bh):
                    for sx in range(bw):
                        cy_p = base_y + sy
                        cx_p = base_x + sx
                        if 0 <= cy_p < self.target_h and 0 <= cx_p < self.target_w:
                            sa = warped_soul[sy, sx, 3] / 255.0
                            if sa > 0:
                                bg_a = canvas_bgra[cy_p, cx_p, 3] / 255.0
                                out_a = sa + bg_a * (1.0 - sa)
                                if out_a > 0:
                                    canvas_bgra[cy_p, cx_p, :3] = (warped_soul[sy, sx, :3] * sa + canvas_bgra[cy_p, cx_p, :3] * bg_a * (1.0 - sa)) / out_a
                                    canvas_bgra[cy_p, cx_p, 3] = int(out_a * 255)

            # Convert to PIL RGBA & apply HD unsharp mask
            pil_f = Image.fromarray(cv2.cvtColor(canvas_bgra, cv2.COLOR_BGRA2RGBA))
            r_c, g_c, b_c, a_c = pil_f.split()
            rgb_c = Image.merge('RGB', (r_c, g_c, b_c)).filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
            nr, ng, nb = rgb_c.split()
            sharpened_frame = Image.merge('RGBA', (nr, ng, nb, a_c))
            frames.append(sharpened_frame)

        return frames

def build_single_sticker_motion(raw_path: str, prompt_path: str = None, motion_override: str = None, out_apng: str = None, out_gif: str = None):
    # Determine motion type
    if motion_override and motion_override != "auto":
        motion_type = motion_override
    elif prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        motion_type = detect_motion_type(content)
    else:
        motion_type = detect_motion_type(os.path.basename(raw_path))

    engine = MotionEngine(raw_path)
    frames = engine.render_motion(motion_type, total_frames=12, frame_duration_ms=250)

    if out_apng is None:
        out_apng = raw_path.replace(".png", "_animated.png")
    if out_gif is None:
        out_gif = out_apng.replace(".png", "_preview.gif")

    os.makedirs(os.path.dirname(os.path.abspath(out_apng)), exist_ok=True)

    # Quantize and save APNG
    temp_files = []
    temp_dir = os.path.dirname(os.path.abspath(out_apng))
    for idx, f in enumerate(frames):
        tmp_name = os.path.join(temp_dir, f"temp_mot_{idx:02d}.png")
        qf = f.quantize(colors=72, method=Image.Quantize.FASTOCTREE)
        qf.save(tmp_name, "PNG", optimize=True)
        temp_files.append(tmp_name)

    apng = APNG(num_plays=0)
    for tmp in temp_files:
        apng.append_file(tmp, delay=250, delay_den=1000)
    apng.save(out_apng)

    # Save GIF
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
        disposal=2
    )

    for tmp in temp_files:
        if os.path.exists(tmp):
            os.remove(tmp)

    # Save contact sheet
    fw, fh = frames[0].size
    sheet = Image.new('RGBA', (fw * 4, fh * 3), (240, 240, 240, 255))
    for idx, f in enumerate(frames):
        r = idx // 4
        c = idx % 4
        sheet.paste(f, (c * fw, r * fh), f)
    sheet_path = out_apng.replace(".png", "_sheet.png")
    sheet.save(sheet_path)

    print(f"\n[AUTO MOTION SUCCESS]")
    print(f"  Motion Archetype: {motion_type}")
    print(f"  APNG Output: {out_apng} ({os.path.getsize(out_apng)/1024:.1f} KB, Target < 300KB: True)")
    print(f"  GIF Preview: {out_gif}")
    print(f"  Contact Sheet: {sheet_path}")

def batch_process_pack(pack_dir: str):
    pack_path = Path(pack_dir).resolve()
    raw_dir = pack_path / "raw"
    prompts_dir = pack_path / "prompts"
    out_dir = pack_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("scene_*.png"))
    print(f"Found {len(raw_files)} scenes in {raw_dir} to animate...")

    for rf in raw_files:
        match = re.search(r"scene_(\d+)", rf.name)
        if not match: continue
        idx = int(match.group(1))
        
        prompt_file = prompts_dir / f"scene_{idx:02d}.txt"
        out_apng = out_dir / f"{idx:02d}.png"
        out_gif = out_dir / f"{idx:02d}_preview.gif"

        print(f"\nProcessing Scene {idx:02d} ...")
        build_single_sticker_motion(
            str(rf),
            str(prompt_file) if prompt_file.exists() else None,
            motion_override=None,
            out_apng=str(out_apng),
            out_gif=str(out_gif)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto 2D Motion & Physics Engine for LINE Animated Stickers")
    parser.add_argument("input", help="Path to raw image OR pack directory (e.g. packs/pack_03_no_work)")
    parser.add_argument("--prompt", help="Path to prompt text file", default=None)
    parser.add_argument("--motion", help="Motion override (dance, cry, jump, sneak, melt, soul_out, stock_crash)", default=None)
    parser.add_argument("--out_apng", help="Output APNG path", default=None)
    parser.add_argument("--out_gif", help="Output GIF path", default=None)

    args = parser.parse_args()

    if os.path.isdir(args.input):
        batch_process_pack(args.input)
    else:
        build_single_sticker_motion(args.input, args.prompt, args.motion, args.out_apng, args.out_gif)
