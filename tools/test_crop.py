import cv2
import numpy as np
import glob
import os

def test_valley_crop(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    col_sums = np.sum(thresh, axis=0)
    
    width = img.shape[1]
    ideal_cuts = [int(width * i / 5) for i in range(1, 5)]
    
    actual_cuts = []
    # Search window around ideal cut
    window = int(width / 5 * 0.3) 
    
    for ideal in ideal_cuts:
        start = max(0, ideal - window)
        end = min(width, ideal + window)
        
        # Find minimum in this window
        window_sums = col_sums[start:end]
        min_idx = np.argmin(window_sums)
        actual_cuts.append(start + min_idx)
        
    print(f"{os.path.basename(input_path)}")
    print(f"  Ideal cuts : {ideal_cuts}")
    print(f"  Actual cuts: {actual_cuts}")
    print(f"  Col sums at cuts: {[col_sums[c] for c in actual_cuts]}")

if __name__ == "__main__":
    artifact_dir = r'C:\Users\lsyan\.gemini\antigravity-ide\brain\10e2f453-8ebb-4681-a920-6f77359938ad'
    sprites = glob.glob(os.path.join(artifact_dir, 'scene_*_sprite_*.png'))
    for s in sprites[:3]:
        test_valley_crop(s)
