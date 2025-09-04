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
    mask_path, rgb_dir, output_dir = args
    
    try:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}")
            return False
        
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        bbox = find_largest_circular_area(mask)
        
        if bbox is None:
            print(f"No circular area found in: {mask_path}")
            return False
        
        base_name = mask_path.stem.replace("_masks", "")
        
        rgb_path = rgb_dir / f"{base_name}.png"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"{base_name}.jpg"
            if not rgb_path.exists():
                rgb_path = rgb_dir / f"{base_name}.jpeg"
                if not rgb_path.exists():
                    print(f"No matching RGB image found for: {base_name}")
                    return False
        
        rgb_img = cv2.imread(str(rgb_path))
        if rgb_img is None:
            print(f"Failed to load RGB image: {rgb_path}")
            return False
        
        x, y, w, h = bbox
        crop = rgb_img[y:y+h, x:x+w]
        
        resized_crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
        
        output_path = output_dir / f"{mask_path.stem}.png"
        cv2.imwrite(str(output_path), resized_crop)
        
        return True
        
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <mask_images_dir> <rgb_images_dir>")
        sys.exit(1)
    
    mask_dir = Path(sys.argv[1])
    rgb_dir = Path(sys.argv[2])
    output_dir = mask_dir / "results_cropped"
    output_dir.mkdir(exist_ok=True)
    
    mask_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        mask_files.extend(mask_dir.glob(f"*{ext}"))
        mask_files.extend(mask_dir.glob(f"*{ext.upper()}"))
    
    if not mask_files:
        print("No mask files found")
        return
    
    args = [(mask_file, rgb_dir, output_dir) for mask_file in mask_files]
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_mask, args)
    
    print(f"Processed {sum(results)}/{len(mask_files)} files")

if __name__ == "__main__":
    main()
