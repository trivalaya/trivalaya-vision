import os
import glob
import json

# Config
DATA_ROOT = "extracted_data"
OUTPUT_FILE = "gallery.html"

HTML_HEADER = """
<!DOCTYPE html>
<html>
<head>
    <title>Trivalaya Vision Gallery</title>
    <style>
        body { background: #1a1a1a; color: #ddd; font-family: sans-serif; padding: 20px; }
        h1 { border-bottom: 1px solid #444; padding-bottom: 10px; }
        .category { margin-bottom: 40px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .card { background: #2a2a2a; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid #333; }
        .card img { max-width: 100%; height: auto; border-radius: 4px; background: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAIklEQVQIW2NkQAKrVq36zwjjgzjwqonyABJwnSYEQPAwAgB94RYW+8x3oAAAAABJRU5ErkJggg=='); }
        .meta { font-size: 0.85em; color: #888; margin-top: 8px; text-align: left; }
        .tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.7em; font-weight: bold; margin-right: 4px; }
        .tag-obv { background: #2c3e50; color: #3498db; border: 1px solid #3498db; }
        .tag-rev { background: #503e2c; color: #e67e22; border: 1px solid #e67e22; }
        .confidence { color: #27ae60; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🏛️ Trivalaya Extraction Gallery</h1>
"""

def generate_gallery():
    html = HTML_HEADER
    
    # Process each category folder
    for category in ["coins", "fragments", "artifacts", "review"]:
        folder = os.path.join(DATA_ROOT, category)
        if not os.path.exists(folder): continue
        
        html += f"<div class='category'><h2>📂 {category.title()}</h2><div class='grid'>"
        
        # Find all JSON files (using them as the index)
        json_files = sorted(glob.glob(os.path.join(folder, "*_meta.json")))
        
        for json_path in json_files:
            # Load Meta
            with open(json_path, 'r') as f:
                meta = json.load(f)
            
            base_name = os.path.basename(json_path).replace("_meta.json", "")
            
            # Find Image (prefer transparent, fallback to crop)
            img_path = os.path.join(folder, f"{base_name}_transparent.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(folder, f"{base_name}_crop.jpg")
                
            # Relative path for HTML
            rel_img_path = os.path.join(category, os.path.basename(img_path))
            
            # Extract metrics
            l1_label = meta['layer_1']['classification']['label']
            l2_label = meta['layer_2']['context_classification']['inferred_container']
            parent = meta.get('provenance', {}).get('parent_image', 'Unknown')
            role = meta.get('provenance', {}).get('inferred_role', 'unknown')
            
            role_tag = ""
            if role == "obverse": role_tag = "<span class='tag tag-obv'>OBV</span>"
            if role == "reverse": role_tag = "<span class='tag tag-rev'>REV</span>"

            html += f"""
            <div class='card'>
                <img src='{rel_img_path}' loading='lazy'>
                <div class='meta'>
                    <strong>{base_name}</strong><br>
                    {role_tag} <span style='color:#bbb'>{parent}</span><br>
                    <hr style='border: 0; border-top: 1px solid #444; margin: 5px 0;'>
                    geom: {l1_label}<br>
                    ctx: {l2_label}
                </div>
            </div>
            """
        
        html += "</div></div>"

    html += "</body></html>"
    
    with open(os.path.join(DATA_ROOT, OUTPUT_FILE), "w") as f:
        f.write(html)
    
    print(f"✅ Gallery generated at: {os.path.join(DATA_ROOT, OUTPUT_FILE)}")

if __name__ == "__main__":
    generate_gallery()