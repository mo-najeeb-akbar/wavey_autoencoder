import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

def find_largest_circular_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_circle = None
    best_score = -1
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
            
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if radius < 10:
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        circle_area = np.pi * radius * radius
        score = circularity * circle_area
        
        if score > best_score:
            best_score = score
            best_circle = (int(x), int(y), int(radius))
    
    if best_circle is None:
        return None
    
    cx, cy, r = best_circle
    x = max(0, cx - r)
    y = max(0, cy - r)
    w = min(mask.shape[1] - x, 2 * r)
    h = min(mask.shape[0] - y, 2 * r)
    
    return (x, y, w, h)

def process_single_mask(args):
    input_path, output_dir = args
    
    try:
        mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        bbox = find_largest_circular_area(mask)
        
        if bbox is None:
            return False
        
        x, y, w, h = bbox
        
        output_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        output_path = output_dir / f"{input_path.stem}_bbox.png"
        cv2.imwrite(str(output_path), output_img)
        
        return True
        
    except:
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_images>")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = input_dir / "results_box"
    output_dir.mkdir(exist_ok=True)
    
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        mask_files.extend(input_dir.glob(f"*{ext}"))
        mask_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not mask_files:
        print("No image files found")
        return
    
    args = [(mask_file, output_dir) for mask_file in mask_files]
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_mask, args)
    
    print(f"Processed {sum(results)}/{len(mask_files)} files")

if __name__ == "__main__":
    main()
