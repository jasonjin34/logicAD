import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

from anomalib.utils.llm import load_gdino_model
from anomalib.models.logicad.sliding_window import (
    dino_image_transform,
    get_bbx_from_query,
    patch_extraction_from_box
)

from anomalib.utils.llm.vlm import vlm_generate_description
from anomalib.utils.llm.base import (
    json2txt,
    txt2embedding,
    cos_sim,
    txt2sum,
    img2text,
    txt2txt
)

from anomalib.models.logicad.text_prompt import (
    TEXT_EXTRACTOR_PROMPTS,
    TEXT_SUMMATION_PROMPTS,
    TEXT_EXTRACTOR_PROMPTS_SA,
    TEXT_SUMMATION_PROMPTS_SA,
)

#path
TRAIN_IMAGE_PATH = "D:/MA/LogicAD/datasets/juice_bottle/train/good/000.png"
TEST_IMAGE_PATH = "D:/MA/LogicAD/datasets/juice_bottle/test/logical_anomaliesd/000.png"
REFERENCE_DESC_PATH = "D:/MA/LogicAD/datasets/juice_bottle/reference_description.txt"
REFERENCE_EMBEDDING_PATH = "D:/MA/LogicAD/datasets/juice_bottle/reference_embedding.npy"
REFERENCE_JSON_PATH = "D:/MA/LogicAD/datasets/juice_bottle/reference_summary.json"

API_KEY_PATH = "D:/MA/LogicAD/keys/gpt.json"
MODEL_EMBEDDING = "text-embedding-3-large-az"   # text-embedding-3-large
CATEGORY = "juice_bottle"                      

def generate_and_save_reference(train_image_path):
    prompt = TEXT_EXTRACTOR_PROMPTS[CATEGORY]
    #reference_description = vlm_generate_description(train_image_path, prompt=prompt)
    """ reference_description = extract_with_GDINO(
        image_path=train_image_path,
        prompt=TEXT_EXTRACTOR_PROMPTS[CATEGORY],
        api_key=API_KEY_PATH,
        model_vlm="gpt-4o-az",
        model_llm="gpt-4o-az",
        category=CATEGORY
    ) """

    reference_description = img2text(
        image=train_image_path,
        api_key=API_KEY_PATH,
        model_name="gpt-4o-az",
        query=prompt,
        temperature=0.2,
        max_tokens=100,
    )

    print(f"Reference description:\n{reference_description}")
    with open(REFERENCE_DESC_PATH, "w", encoding="utf-8") as f:
        f.write(reference_description)
    print(f"Saved reference description to {REFERENCE_DESC_PATH}")
    
    #Summarization
    summary_prompt = TEXT_SUMMATION_PROMPTS[CATEGORY]
    summary_json = txt2sum(
        input_text=reference_description,
        few_shot_message=summary_prompt,
        api_key=API_KEY_PATH,
        model="gpt-4o-az"
    )
    print("reference summary json:", summary_json)
    with open(REFERENCE_JSON_PATH, "w", encoding="utf-8") as f:
        f.write(summary_json)

    #convert to embedding
    structured_text = json2txt(summary_json,values_only=True)
    embedding_ref = txt2embedding(
        input_text= structured_text,
        api_key=API_KEY_PATH,
        model=MODEL_EMBEDDING
    )
    embedding_ref = np.array(embedding_ref)

    np.save(REFERENCE_EMBEDDING_PATH, embedding_ref)
    print(f"Saved reference embedding to {REFERENCE_EMBEDDING_PATH}")
    return embedding_ref, summary_json


def test_inference(test_image_path, reference_embedding):

    prompt = TEXT_EXTRACTOR_PROMPTS[CATEGORY]
    #test_description = vlm_generate_description(test_image_path, prompt=prompt)
    """ test_description = extract_with_GDINO(
        image_path=test_image_path,
        prompt=TEXT_EXTRACTOR_PROMPTS[CATEGORY],
        api_key=API_KEY_PATH,
        model_vlm="gpt-4o-az",
        model_llm="gpt-4o-az",
        category=CATEGORY
    ) """

    test_description = img2text(
        image=test_image_path,
        api_key=API_KEY_PATH,
        model_name="gpt-4o-az", 
        query=prompt,
        temperature=0.2,
        max_tokens=100,
    )

    print(f"Test description:\n{test_description}")

    summary_prompt = TEXT_SUMMATION_PROMPTS[CATEGORY]
    summary_json = txt2sum(
        input_text= test_description,
        few_shot_message=summary_prompt,
        api_key=API_KEY_PATH,
        model="gpt-4o-az"
    )
    print("summary_json:",summary_json)

    test_text = json2txt(summary_json,values_only=True)
    test_embedding = txt2embedding(
        input_text=test_text,
        api_key=API_KEY_PATH,
        model=MODEL_EMBEDDING
    )
    test_embedding = np.array(test_embedding)

    #calculate anomalib scores
    score = 1 - cos_sim(reference_embedding, test_embedding)
    print(f"Anomalib Score = {score:.4f}")

    #compare JSON
    diffs = compare_json_values(summary_json_train, summary_json)
    if diffs:
        print("Mismatched Fields:")
        for d in diffs:
            print("  ", d)
    else:
        print("All fields match.")

    diffs = compare_json_values(summary_json_train, summary_json)  
    #visualization
    create_comparison_visual(TRAIN_IMAGE_PATH, test_image_path, score, diffs)

    return score

def compare_json_values(train_json: str, test_json: str) -> list[str]:
    def normalize_key(key):
        return key.strip().replace(" ", "_").replace("-", "_").lower()

    try:
        train_data = json.loads(train_json)
        test_data = json.loads(test_json)
    except json.JSONDecodeError:
        return ["Invalid JSON format."]

    train_key_map = {normalize_key(k): k for k in train_data}
    test_key_map = {normalize_key(k): k for k in test_data}

    mismatches = []
    for norm_key in train_key_map:
        if norm_key in test_key_map:
            k1 = train_key_map[norm_key]
            k2 = test_key_map[norm_key]
            v1 = str(train_data[k1]).strip().lower()
            v2 = str(test_data[k2]).strip().lower()
            if v1 != v2:
                mismatches.append(f'"{norm_key}": train="{train_data[k1]}", test="{test_data[k2]}"')
    return mismatches


def create_comparison_visual(
    train_img_path: str,
    test_img_path: str,
    anomalib_score: float,
    diff_texts: list[str],
    figsize=(12, 8) 
):
    """
    Display train/test images + Anomalib Score + all field differences
    """
    if not os.path.exists(train_img_path):
        raise FileNotFoundError(f"Train image not found: {train_img_path}")
    if not os.path.exists(test_img_path):
        raise FileNotFoundError(f"Test image not found: {test_img_path}")
    
    train_img = Image.open(train_img_path).convert("RGB")
    test_img = Image.open(test_img_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(train_img)
    axes[0].set_title("Train Image")
    axes[0].axis("off")

    axes[1].imshow(test_img)
    axes[1].set_title("Test Image")
    axes[1].axis("off")

    diff_string = "\n".join(diff_texts) if diff_texts else "All fields match."
    display_text = f"Anomalib Score = {anomalib_score:.4f}\n\n Differences:\n{diff_string}"

    plt.subplots_adjust(bottom=0.3)  #Reserve enough space
    fig.text(0.5, 0.02, display_text, ha='center', va='bottom', fontsize=10, wrap=True)

    plt.show()

def extract_with_GDINO(
    image_path: str,
    prompt: str,
    api_key: str,
    model_vlm: str,
    model_llm: str,
    temperature: float = 0.2,
    max_tokens: int = 100,
    img_size: int = 224,
    top_p: float = None,
    seed: int = 42,
    category: str = "juice_bottle"
) -> str:
    """
    Use GroundingDINO to extract patches
    """
    gdino = load_gdino_model(ckpt_path=None, cfg="swint")
    raw_img, trans_img = dino_image_transform(image_path)
    boxes, _, _ = get_bbx_from_query(trans_img, gdino, query=category)
    patches = patch_extraction_from_box(raw_img, boxes, patch=1)

    text = ""
    for p in patches:
        patch_text_list = []
        for _ in range(1):
            patch_text = img2text(
                image=p,
                api_key=api_key,
                model_name=model_vlm,
                query=prompt,
                max_tokens=max_tokens,
                img_size=img_size,
                temperature=temperature,
                top_p=top_p,
                seed=seed
            )
            patch_text_list.append(patch_text)
        test_list_str = str(patch_text_list)
        print("test_list_str:",test_list_str)
        text_path = txt2txt(
            test_list_str,
            query="select the most frequent text from the list, only give the text",
            api_key=api_key,
            model=model_llm
        ).replace('"', '')
        text += "patch description: " + text_path + "\n"

    if category not in ["screw_bag", "splicing_connectors"]:
        text = txt2txt(
            text,
            query="select the unique text from the list, only give the unique text",
            api_key=api_key,
            model=model_llm
        ).replace('"', '')
    return text



if __name__ == "__main__":

    if not os.path.exists(REFERENCE_EMBEDDING_PATH):
        reference_embedding, summary_json_train = generate_and_save_reference(TRAIN_IMAGE_PATH)
    else:
        reference_embedding = np.load(REFERENCE_EMBEDDING_PATH)
        with open(REFERENCE_DESC_PATH.replace("reference_description.txt", "reference_summary.json"), "r", encoding="utf-8") as f:
            summary_json_train = f.read()

    test_inference(TEST_IMAGE_PATH, reference_embedding)
