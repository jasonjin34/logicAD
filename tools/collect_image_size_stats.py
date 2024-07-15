from pathlib import Path
import csv
import cv2
import json
import sys
from tqdm import tqdm


root_dir = Path(sys.argv[1])
split_file = root_dir / "split_csv" / "1cls.csv"

category_stats = {}

try:
    with split_file.open(encoding="utf-8") as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in tqdm(csvreader):
            category, split, label, image_path, mask_path = row
            image_name = image_path.split("/")[-1]
            image_path = root_dir / image_path
            image = cv2.imread(str(image_path))
            image_size = image.shape[:-1]
            if category not in category_stats or image_size not in category_stats[category]:
                category_stats.setdefault(category, {})
                category_stats[category][image_size] = 1
            else:
                category_stats[category][image_size] += 1
except:
    pass

with open(str(root_dir.name) + ".json", "w") as f:
    category_stats = {ck: {str(k): v for k, v in cv.items()} for ck, cv in category_stats.items()}
    json_str = json.dumps(category_stats, indent=4, sort_keys=True)
    f.write(json_str)
