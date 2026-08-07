import os
import argparse
import cv2
import numpy as np
from PIL import Image
from apng import APNG

def remove_white_background(img):
    """
    Flood fills the white background from the edges to make it transparent.
    img is a cv2 BGRA image.
    """
    h, w = img.shape[:2]
    bgr = img[:, :, :3].copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    
    tolerance = (20, 20, 20)
    corners = [(0,0), (w-1,0), (0,h-1), (w-1,h-1), (w//2, 0), (w//2, h-1)]
    for pt in corners:
        bg_color = bgr[pt[1], pt[0]]
        if bg_color[0] > 230 and bg_color[1] > 230 and bg_color[2] > 230:
            cv2.floodFill(bgr, mask, pt, (255, 0, 255), tolerance, tolerance, cv2.FLOODFILL_FIXED_RANGE)
            
    filled_mask = mask[1:-1, 1:-1]
    img[filled_mask == 1, 3] = 0
    return img

def process_single_image(input_path, output_path, total_frames=5, duration_ms=200):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load {input_path}")
        return
        
    has_alpha = False
    if img.shape[2] == 4:
        if np.any(img[:, :, 3] < 255):
            has_alpha = True

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    if not has_alpha:
        img = remove_white_background(img)
    
    # Find bounding box of content
    alpha = img[:, :, 3]
    y_idx, x_idx = np.where(alpha > 0)
    if len(y_idx) == 0:
        return
        
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min, y_max = np.min(y_idx), np.max(y_idx)
    
    # Crop to content
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA))
    cw, ch = cropped_pil.size
    
    # LINE rules: max 320x270, one side must be exactly 270.
    # We will animate a vertical bounce. Max jump = 15 pixels.
    # So max content height should be 270 - 15 = 255.
    max_w = 320
    max_h = 255
    
    scale = min(max_w / cw, max_h / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    resized = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Define canvas size
    canvas_w = min(320, max(270, new_w))
    canvas_h = 270
    if new_w <= 270 and new_h <= 270:
        canvas_w = 270
        canvas_h = 270
    elif new_w > 270:
        canvas_w = new_w
        canvas_h = 270
        
    # Animation offsets (bounce up and down)
    offsets_y = [0, -15, -5, 0, 0] # for 5 frames
    if len(offsets_y) != total_frames:
        offsets_y = [0] * total_frames
        
    processed_frames = []
    
    for i in range(total_frames):
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        offset_x = (canvas_w - new_w) // 2
        # Default ground align, plus dynamic animation offset
        base_y = canvas_h - new_h
        offset_y = base_y + offsets_y[i]
        
        canvas.paste(resized, (offset_x, offset_y), resized)
        processed_frames.append(canvas)
        
    temp_files = []
    for i, frame in enumerate(processed_frames):
        tmp_name = f"temp_frame_{i}.png"
        frame.save(tmp_name, "PNG")
        temp_files.append(tmp_name)
        
    total_time = duration_ms * total_frames
    if total_time > 4000:
        duration_ms = 4000 // total_frames
        
    apng = APNG()
    for tmp_name in temp_files:
        apng.append_file(tmp_name, delay=duration_ms, delay_den=1000)
        
    apng.save(output_path)
    
    for tmp_name in temp_files:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
            
    print(f"Generated {output_path} with {len(processed_frames)} frames (canvas: {canvas_w}x{canvas_h}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build animated LINE APNG from Single Image")
    parser.add_argument("input", help="Input single image")
    parser.add_argument("output", help="Output APNG file")
    
    args = parser.parse_args()
    process_single_image(args.input, args.output)
