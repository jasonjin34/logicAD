import os, glob, json
import numpy as np
from numpy import dot
from numpy.linalg import norm
import argparse
from tqdm import tqdm

from anomalib.utils.llm import cos_sim, img2text, txt2sum, txt2embedding, load_gdino_model
from anomalib.models.logicad.text_prompt import TEXT_SUMMATION_PROMPTS, TEXT_EXTRACTOR_PROMPTS
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score

# GLOBAL VARIABLES
API = "/home/erjin/git/Logic_AD_LLM/keys/gpt.json"
LOCO_DATASET_mini = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini"
LOCO_DATASET_origin = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original"


#################### help function for evaluation ########################
def init_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)
            return {}
    else:
        with open(path, "r") as f:
            return json.load(f)


def update_json(path, dict):
    with open(path, "w") as f:
        json.dump(dict, f)


# help function for grabing the dataaset per category for mvtec loco
def calculate_f1_max(y, pred):
    precision, recall, thresholds = precision_recall_curve(y, pred)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    return max_f1


def calculate_auroc(y, pred, pos_label=1):
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def cos_sim(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


# help functions for fast testing and ablation runnings
def get_dataset_per_category(dataset, category, k_shot=1):
    def get_path_list(dir_path):
        image_list = glob.glob(os.path.join(dir_path, "*"))
        if "good" in image_list[0]:
            label = 0
        else:
            label = 1
        label_list = [label] * len(image_list)
        image_list = sorted(image_list, key=lambda x: int(os.path.basename(x)[:3]), reverse=False)
        return image_list, label_list

    train_img = os.path.join(dataset, category, "train", "good")
    test_good, test_ab = os.path.join(dataset, category, "test", "good"), os.path.join(
        dataset, category, "test", "logical_anomalies"
    )
    reference_img, _ = get_path_list(train_img)
    reference_img = reference_img[:k_shot]
    test_good_img, test_good_label = get_path_list(test_good)
    test_ab_img, test_ab_label = get_path_list(test_ab)
    output_dict = {
        "reference_img": reference_img,
        "test_img": test_good_img + test_ab_img,
        "test_label": test_good_label + test_ab_label,
    }
    return output_dict


def text_extraction(p, query, api=API):
    return img2text(
        p,
        api,
        img_size=None,
        query=query,
        temperature=0.05,
        top_p=0.05,
        max_tokens=500,
        model_name="gpt-4o-az",
        model=None,
    )["choices"][0]["message"]["content"]


def generate_text_embedding(
    img, 
    query, 
    save_path,
    img2text_dict=None,
    summation_prompt="",
    verbose=True,
):
    if img2text_dict is None:
        img2text_dict = init_json(save_path)

    if isinstance(img2text_dict, str):
        with open(img2text_dict, "r") as f:
            img2text_dict = json.load(f)

    global API
    if img in img2text_dict:
        text = img2text_dict[img]
    else:
        text = text_extraction(img, query)
        img2text_dict[img] = text
        update_json(save_path, img2text_dict)

    text = txt2sum(api_key=API, input_text=text, few_shot_message=summation_prompt, model="gpt-4o-az")
    if verbose:
        print(f"image: {img}, descrptions summary: {text}")
    embedding = txt2embedding(api_key=API, input_text=text, model="text-embedding-3-large-az")
    return embedding


def evaluation(
    imgs, 
    label, 
    ref_embedding, 
    query, 
    save_path,
    summation_prompt="",
    verbose=True,
):
    pred = []
    print("start evaluation ...")
    for img in tqdm(imgs):
        embedding = generate_text_embedding(img=img, query=query, verbose=verbose, save_path=save_path, summation_prompt=summation_prompt)
        score = cos_sim(embedding, ref_embedding)
        pred.append(score)
    pred = [1 - p for p in pred]
    pred = np.array(pred)
    auroc = calculate_auroc(label, pred)
    f1_max = calculate_f1_max(label, pred)
    return auroc, f1_max


def main(args):
    summation_prompt = TEXT_SUMMATION_PROMPTS[args.category]
    query = TEXT_EXTRACTOR_PROMPTS[args.category]
    # get the category img2text extraction file path
    save_path = os.path.join(args.dataset, args.category, args.save_path)
    output = get_dataset_per_category(args.dataset, args.category, args.k_shot)
    imgs, label, ref = output["test_img"], output["test_label"], output["reference_img"]
    ref_embedding = generate_text_embedding(ref[0], query, verbose=args.verbose, save_path=save_path)
    auroc, f1_max = evaluation(imgs=imgs, label=label, ref_embedding=ref_embedding, query=query, verbose=args.verbose, save_path=save_path, summation_prompt=summation_prompt)
    print(auroc, f1_max)


if __name__ == "__main__":
    args = argparse.ArgumentParser("Evaluation for the logic AD fast evalutation pipeline")
    args.add_argument("--dataset", type=str, default=LOCO_DATASET_mini, help="The dataset path for evaluation")
    args.add_argument("--category", type=str, default="splicing_connectors", help="The category for evaluation")
    args.add_argument("--save_path", type=str, default="img_2_text.json", help="The category for evaluation")
    args.add_argument("--k_shot", type=int, default=1, help="The number of reference images for each category")
    args.add_argument("--verbose", type=bool, default=True, help="The flag for verbose output")
    # args.add_argument("--query", type=str, default="describe this image", help="The query for the image")
    args = args.parse_args()
    main(args)
