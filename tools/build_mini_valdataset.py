from pathlib import Path
import csv
import cv2
import json
import os
import shutil
import sys
from tqdm import tqdm

import math, random
random.seed(42)

'''
Build new dataset from the complete dataset based on percentage.
Category imbalance is eliminated
MVTec Test set stats
{
    "bottle": {
        "bad": 63,
        "good": 20,
        "total": 83
    },
    "cable": {
        "bad": 92,
        "good": 58,
        "total": 150
    },
    "capsule": {
        "bad": 109,
        "good": 23,
        "total": 132
    },
    "carpet": {
        "bad": 89,
        "good": 28,
        "total": 117
    },
    "grid": {
        "bad": 57,
        "good": 21,
        "total": 78
    },
    "hazelnut": {
        "bad": 70,
        "good": 40,
        "total": 110
    },
    "leather": {
        "bad": 92,
        "good": 32,
        "total": 124
    },
    "metal_nut": {
        "bad": 93,
        "good": 22,
        "total": 115
    },
    "pill": {
        "bad": 141,
        "good": 26,
        "total": 167
    },
    "screw": {
        "bad": 119,
        "good": 41,
        "total": 160
    },
    "tile": {
        "bad": 84,
        "good": 33,
        "total": 117
    },
    "toothbrush": {
        "bad": 30,
        "good": 12,
        "total": 42
    },
    "transistor": {
        "bad": 40,
        "good": 60,
        "total": 100
    },
    "wood": {
        "bad": 60,
        "good": 19,
        "total": 79
    },
    "zipper": {
        "bad": 119,
        "good": 32,
        "total": 151
    }
}

'''

root_dir = Path(sys.argv[1])
dest_dir = Path(sys.argv[2])
split_file = root_dir / "split_csv" / "1cls.csv"

category_stats = {}

if split_file.exists():
    raise NotImplementedError
    #try:
    #    with split_file.open(encoding="utf-8") as file:
    #        csvreader = csv.reader(file)
    #        next(csvreader)
    #        for row in tqdm(csvreader):
    #            category, split, label, image_path, mask_path = row
    #            image_name = image_path.split("/")[-1]
    #            image_path = root_dir / image_path
    #            image = cv2.imread(str(image_path))
    #            image_size = image.shape[:-1]
    #            if category not in category_stats \
    #                or image_size not in category_stats[category]:
    #                category_stats.setdefault(category, {})
    #                category_stats[category][image_size] = 1
    #            else:
    #                category_stats[category][image_size] += 1
    #except:
    #    # for debugging -- would just skip is interrupted and produce output json anyways
    #    pass
else: # mvtec
    try:
        categories = [c for c in root_dir.iterdir() if c.is_dir()]
        for c in tqdm(categories): # per category 40 images
             test_dir = c / "test"
             category_stats[c.name] = dict()
             for gtd in test_dir.iterdir(): # good, broken_small, contamination... 
                  category_stats[c.name][gtd.name] = []
                  for file in gtd.iterdir():
                       assert file.suffix == ".png"
                       category_stats[c.name][gtd.name].append(file)
    except:
        # for debugging -- would just skip is interrupted and produce output json anyways
        pass
             
def copy_files(category, label, chosen_files):
    stats = []
    (dest_dir / category / "test" / label).mkdir(parents=True)
    if label != "good":
        (dest_dir / category / "ground_truth" / label).mkdir(parents=True)
    for f in chosen_files:
         file_name = f.name
         dest_img_path = dest_dir / category / "test" / label / file_name 
         shutil.copy2(f, dest_img_path)
         stats += [str(dest_img_path)]
         
         if label != "good":
             src_mask_path = root_dir / category / "ground_truth" / label / file_name.replace(".png", "_mask.png") 
             dest_mask_path = dest_dir / category / "ground_truth" / label / file_name.replace(".png", "_mask.png")
             shutil.copy2(src_mask_path, dest_mask_path)
    return stats

output_stats = dict() # {category: {label: files} }
for c in category_stats.keys():
     output_stats[c] = dict() 
     category_total = sum([len(files) for files in category_stats[c].values()])
     category_common_prob = 40 / category_total
     all_chosen_files = []
     for l in sorted(category_stats[c].keys()):
         if l == "good":
             continue
         cur_count = category_common_prob * len(category_stats[c][l])
         chosen_files = random.sample(category_stats[c][l], k=int(math.ceil(cur_count)))
         print(
            f"For category {c} and label {l}, {cur_count} from {len(category_stats[c][l])} files are selected"
         )
         output_stats[c][l] = copy_files(c, l, chosen_files)
         all_chosen_files.extend(chosen_files)
     print(f"For category {c} and label good, {40-len(all_chosen_files)} from {len(category_stats[c]['good'])} files are selected")
     chosen_files = random.sample(category_stats[c]["good"], k=40-len(all_chosen_files))
     output_stats[c]["good"] = copy_files(c, "good", chosen_files)

     src_train_dir = (root_dir / c / "train").absolute()
     dest_train_dir = (dest_dir / c / "train").absolute()
     os.symlink(src_train_dir, dest_train_dir)


with open(str(dest_dir.name)+".json", "w") as f:
    json_str = json.dumps(output_stats, indent=4, sort_keys=True)
    f.write(json_str)
