from PIL import Image
import numpy as np
import os

src_dir = "D:\MA\LogicAD\datasets\juice_bottle\ground_truth\logical_anomalies"
dst_dir = "D:\MA\LogicAD\datasets/juice_bottle/ground_truth/logical_anomalies_merged"

os.makedirs(dst_dir, exist_ok=True)

for subdir in os.listdir(src_dir):
    sub_path = os.path.join(src_dir, subdir)
    if os.path.isdir(sub_path):
        merged_mask = None
        for fname in os.listdir(sub_path):
            mask_path = os.path.join(sub_path, fname)
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            merged_mask = mask if merged_mask is None else (merged_mask | mask)

        if merged_mask is not None:
            merged_mask = (merged_mask * 255).astype(np.uint8)
            Image.fromarray(merged_mask).save(os.path.join(dst_dir, f"{subdir}.png"))
            print(f"Merge finished: {subdir}.png")
