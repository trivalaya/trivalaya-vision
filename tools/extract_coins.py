import cv2
import numpy as np
import os
import glob
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline_manager import analyze_image

# --- CONFIG ---
INPUT_FOLDER = "data/test_images"
OUTPUT_ROOT = "extracted_data"
EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.JPG"]

CLASS_MAP = {
    "Round Object (Coin/Medallion)": "coins",
    "Round Object (Coin)": "coins",
    "Geometric Form": "fragments",
    "Fragment": "fragments",
    "Artifact (Isolated)": "artifacts",
    "Unknown": "review"
}

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def save_json(data, path):
    def convert(o):
        if isinstance(o, (np.int64, np.int32)): return int(o)
        if isinstance(o, np.float32): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    with open(path, 'w') as f: json.dump(data, f, default=convert, indent=4)

def extract_and_save(original_path, detection_list):
    filename = os.path.basename(original_path)
    name_base, ext = os.path.splitext(filename)
    
    full_img = cv2.imread(original_path)
    if full_img is None: return

    # Handle PNG alpha
    if original_path.lower().endswith(".png"):
        img_alpha = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
        if img_alpha is not None and img_alpha.shape[2] == 4:
            alpha = img_alpha[:,:,3]
            full_img[alpha == 0] = [255, 255, 255]

    h_orig, w_orig = full_img.shape[:2]
    MAX_DIMENSION = 3200
    scale_factor = 1.0
    if max(h_orig, w_orig) > MAX_DIMENSION:
        scale_factor = MAX_DIMENSION / float(max(h_orig, w_orig))

    # --- LOGIC: Sort by X Position ---
    def get_center_x(d):
        if "contour" not in d['layer_1']: return 0
        M = cv2.moments(d['layer_1']['contour'])
        if M["m00"] != 0: return int(M["m10"] / M["m00"])
        return 0

    detection_list.sort(key=get_center_x)

    for i, result in enumerate(detection_list):
        final_class = result.get('final_classification', 'Unknown')
        subfolder = CLASS_MAP.get(final_class, "review")
        output_dir = os.path.join(OUTPUT_ROOT, subfolder)
        ensure_dir(output_dir)
        
        # --- NAMING CONVENTION ---
        # If exactly 2 objects found -> assume [Obverse, Reverse]
        suffix = ""
        role = "unknown"
        
        if len(detection_list) == 2:
            if i == 0: 
                suffix = "_obv"
                role = "obverse"
            else: 
                suffix = "_rev"
                role = "reverse"
        elif len(detection_list) > 1:
            suffix = f"_{i+1:02d}"
            
        print(f"   -> Processing '{name_base}{suffix}' as [{subfolder}]...")

        if "contour" not in result['layer_1']: continue
        
        contour_small = result['layer_1']['contour']
        contour_orig = (contour_small / scale_factor).astype(np.int32)

        # Crop
        x, y, w, h = cv2.boundingRect(contour_orig)
        margin = int(max(w, h) * 0.05)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_orig, x + w + margin)
        y2 = min(h_orig, y + h + margin)
        
        crop_img = full_img[y1:y2, x1:x2]
        crop_path = os.path.join(output_dir, f"{name_base}{suffix}_crop.jpg")
        cv2.imwrite(crop_path, crop_img)

        # Transparent
        mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.drawContours(mask, [contour_orig], -1, 255, -1)
        mask_crop = mask[y1:y2, x1:x2]
        b, g, r = cv2.split(crop_img)
        transparent_img = cv2.merge([b, g, r, mask_crop])
        trans_path = os.path.join(output_dir, f"{name_base}{suffix}_transparent.png")
        cv2.imwrite(trans_path, transparent_img)

        # Save Meta
        result['provenance'] = {
            "parent_image": filename,
            "inferred_role": role
        }
        meta_path = os.path.join(output_dir, f"{name_base}{suffix}_meta.json")
        save_json(result, meta_path)

def run_extraction_batch():
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    files.sort()
    
    print(f"🚀 Starting Extraction with Scene Awareness...")
    
    for file_path in files:
        print(f"Analyzing {os.path.basename(file_path)}...")
        result = analyze_image(file_path)
        
        if result['status'] == 'success':
            extract_and_save(file_path, result['detections'])
        else:
            print(f"   ⛔ No Objects Found.")

if __name__ == "__main__":
    run_extraction_batch()