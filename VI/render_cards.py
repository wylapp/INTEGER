import os
from PIL import Image
import argparse
from tqdm import tqdm
import base64
import re
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from playwright.sync_api import sync_playwright
import pandas as pd

DATASET = 'Arts_Crafts_and_Sewing'
DATA_FOLDER = 'data_18'
meta_info = pd.read_parquet(os.path.join("..", DATA_FOLDER, DATASET, "meta.parquet"))
env = Environment(loader=FileSystemLoader('.'))
os.makedirs(os.path.join("..", DATA_FOLDER, DATASET, "cards_raw"), exist_ok=True)
def resize_images(input_path, output_path, target_size=(384, 128)):
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"creating output directory: {output_path}")
    
    processed_count = 0
    failed_count = 0
    
    for filename in tqdm(os.listdir(input_path)):
        if filename.lower().endswith(supported_formats):
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            
            try:
                with Image.open(input_file) as img:
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    resized_img.save(output_file)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Failed in resizing {filename}: {str(e)}")
                failed_count += 1

def encode_image_to_base64(img_path: Path) -> tuple:
    if not img_path.is_file():
        # change to default image if cover image not found
        img_path = Path("images/Notfound.png")
    ext = img_path.suffix.lower().lstrip('.')
    if ext == 'jpg':
        ext = 'jpeg'
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return ext, b64

def build_html(thumb_path: str, title: str, labels: str, abs: str, price) -> str:
    template = env.get_template("item_entry_template.html")
    img_ext, img_b64 = encode_image_to_base64(Path(thumb_path))
    return template.render(
        img_ext=img_ext,
        img_base64=img_b64,
        title=title,
        labels=labels,
        abs = abs,
        price=price
    )


def render_to_png(viewport=(336, 672)):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]},
            device_scale_factor=1,
        )
        page = context.new_page()
        for _, line in tqdm(meta_info.iterrows()):
            if os.path.exists(f"../{DATA_FOLDER}/{DATASET}/cards_raw/{line['asin']}.jpg"):
                continue
            
            labels = line["categories"].replace('</span>',"").split(">")
            if line["brand"] != "Unknown":
                labels = [line["brand"]] + labels
            if line["price"] == "Data Unavailable":
                price = None
            else:
                price = re.findall(r"[0-9]+", line["price"])
            html_page = build_html(thumb_path=f"../{DATA_FOLDER}/{DATASET}/images/{line['asin']}.jpg", 
                                   title=line["title"], 
                                   labels=labels,
                                   abs=line["description"], price=price)
            page.set_content(html_page)
            page.wait_for_timeout(200)

            page.screenshot(type="jpeg", path=f"../{DATA_FOLDER}/{DATASET}/cards_raw/{line['asin']}.jpg", quality=85, full_page=False)

        browser.close()

render_to_png()
resize_images(f"../{DATA_FOLDER}/{DATASET}/cards_raw", f"../{DATA_FOLDER}/{DATASET}/cards", (336, 672))