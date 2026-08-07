import cv2
import numpy as np

def test_contours(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 20: # filter noise
            boxes.append((x, y, w, h))
            
    boxes.sort(key=lambda b: b[0])
    print(f"Found {len(boxes)} major bounding boxes in {img_path}.")
    for b in boxes:
        print(f"  Box: x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}")

test_contours(r"C:\Users\lsyan\.gemini\antigravity-ide\brain\10e2f453-8ebb-4681-a920-6f77359938ad\scene_01_test_1783781547070.png")
