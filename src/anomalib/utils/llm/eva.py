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

# Path config
NORMAL_DIR = "D:/MA/LogicAD/datasets/juice_bottle/train/good"
ANOMALY_DIR = "D:/MA/LogicAD/datasets/juice_bottle/test/logical_anomalies"

THRESHOLD = 0.03  # Anomalib Score threshold

# Load reference embedding and summary
reference_embedding = np.load(REFERENCE_EMBEDDING_PATH)
with open(REFERENCE_DESC_PATH.replace("reference_description.txt", "reference_summary.json"), "r", encoding="utf-8") as f:
    summary_json_train = f.read()

# Storage
scores = []
predictions = []
labels = []  # 0 = normal, 1 = anomaly

def collect_scores_from_folder(folder_path, label):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_path in tqdm(image_files, desc=f"Evaluating {'Anomaly' if label == 1 else 'Normal'}"):
        try:
            score = test_inference(img_path, reference_embedding, summary_json_train)
            scores.append(score)
            predictions.append(1 if score > THRESHOLD else 0)
            labels.append(label)
        except Exception as e:
            print(f"⚠️ Failed processing {img_path}: {e}")

if __name__ == "__main__":
    # Process normal images (label = 0)
    collect_scores_from_folder(NORMAL_DIR, label=0)

    # Process anomaly images (label = 1)
    collect_scores_from_folder(ANOMALY_DIR, label=1)

    # Compute metrics
    f1 = f1_score(labels, predictions)
    auroc = roc_auc_score(labels, scores)

    print("\n====== Evaluation Results ======")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("================================")
