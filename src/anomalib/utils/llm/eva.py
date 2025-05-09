""" import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from demo import (
    test_inference,
    REFERENCE_EMBEDDING_PATH,
    REFERENCE_DESC_PATH,
    CATEGORY,
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
    json2txt,
)
import json
from anomalib.utils.llm.vlm import load_llava_custom  

# Path config
NORMAL_DIR = "/home/students/yuehan/projects/logicAD/datasets/juice_bottle/train/good"
ANOMALY_DIR = "/home/students/yuehan/projects/logicAD/datasets/juice_bottle/test/logical_anomalies"

THRESHOLD = 0.015  # Anomalib Score threshold

# Load reference embedding and summary
reference_embedding = np.load(REFERENCE_EMBEDDING_PATH)
with open(REFERENCE_DESC_PATH.replace("reference_description.txt", "reference_summary.json"), "r", encoding="utf-8") as f:
    summary_json_train = f.read()

# Storage
scores = []
predictions = []
labels = []  # 0 = normal, 1 = anomaly

    
global_head_mask = None  # 新增，全局缓存 mask

def collect_scores_from_folder(folder_path, label, llava_model, tokenizer, image_processor):
    global global_head_mask  # 共享缓存
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_path in tqdm(image_files, desc=f"Evaluating {'Anomaly' if label == 1 else 'Normal'}"):
        try:
            score, global_head_mask = test_inference(img_path, reference_embedding, summary_json_train, head_mask=global_head_mask,llava_model=llava_model, tokenizer=tokenizer, image_processor=image_processor)
            scores.append(score)
            predictions.append(1 if score > THRESHOLD else 0)
            labels.append(label)

        except Exception as e:
            print(f"⚠️ Failed processing {img_path}: {e}")

if __name__ == "__main__":
    tokenizer, llava_model, image_processor, _ = load_llava_custom(model_path="liuhaotian/llava-v1.5-7b")

    # Process normal images (label = 0)
    collect_scores_from_folder(NORMAL_DIR, label=0,llava_model=llava_model, tokenizer=tokenizer, image_processor=image_processor)

    # Process anomaly images (label = 1)
    collect_scores_from_folder(ANOMALY_DIR, label=1,llava_model=llava_model, tokenizer=tokenizer, image_processor=image_processor)

    # Compute metrics
    f1 = f1_score(labels, predictions)
    auroc = roc_auc_score(labels, scores)

    print("\n====== Evaluation Results ======")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("================================")
 """

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from demo import (
    test_inference,
    REFERENCE_EMBEDDING_PATH,
    REFERENCE_DESC_PATH,
    CATEGORY,
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
    json2txt,
)
import json
from anomalib.utils.llm.vlm import load_llava_custom

# Path config
NORMAL_DIR = "/home/students/yuehan/projects/logicAD/datasets/juice_bottle/train/good"
ANOMALY_DIR = "/home/students/yuehan/projects/logicAD/datasets/juice_bottle/test/logical_anomalies"

THRESHOLD = 0.015  # Anomalib Score threshold

# Load reference embedding and summary
reference_embedding = np.load(REFERENCE_EMBEDDING_PATH)
with open(REFERENCE_DESC_PATH.replace("reference_description.txt", "reference_summary.json"), "r", encoding="utf-8") as f:
    summary_json_train = f.read()

# Storage
scores = []
predictions = []
labels = []  # 0 = normal, 1 = anomaly

def collect_scores_from_folder(folder_path, label, llava_model, tokenizer, image_processor):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_path in tqdm(image_files, desc=f"Evaluating {'Anomaly' if label == 1 else 'Normal'}"):

        score = test_inference(
            img_path,
            reference_embedding,
            summary_json_train,
            llava_model=llava_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )
        scores.append(score)
        predictions.append(1 if score > THRESHOLD else 0)
        labels.append(label)

if __name__ == "__main__":
    tokenizer, llava_model, image_processor, _ = load_llava_custom(model_path="liuhaotian/llava-v1.5-7b")
    
    # Process normal images (label = 0)
    collect_scores_from_folder(
        NORMAL_DIR,
        label=0,
        llava_model=llava_model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # Process anomaly images (label = 1)
    collect_scores_from_folder(
        ANOMALY_DIR,
        label=1,
        llava_model=llava_model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )

    # Compute metrics
    f1 = f1_score(labels, predictions)
    auroc = roc_auc_score(labels, scores)

    print("\n====== Evaluation Results ======")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("================================")
