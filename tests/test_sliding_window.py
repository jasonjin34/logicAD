import torch

from anomalib.models.logicad.sliding_window import dino_image_transform
from anomalib.models.logicad.sliding_window import dino_image_transform, patch_extraction_from_box, get_bbx_from_query
from anomalib.utils.llm import img2text, txt2sum, load_gdino_model
from anomalib.utils.visualization import display_image, display_multiple_image, image_unfold, load_image


def test_dino_image_transform():
    path = "tests/test_data/few_pushpin.png"
    image, transformed_img = dino_image_transform(path)
    assert image is not None
    assert transformed_img is not None


def test_patch_text_extraction():
    category = "pushpin"
    api_key = "/home/erjin/git/Logic_AD_LLM/keys/gpt.json"
    path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/mini/pushpins/test/logical_anomalies/023.png"

    img, trans_img = dino_image_transform(path)
    model = load_gdino_model(ckpt_path="/home/erjin/git/Logic_AD_LLM/datasets/GroundingDino/groundingdino_swint_ogc.pth")
    boxes, _, _ = get_bbx_from_query(image=trans_img, model=model)
    patches = patch_extraction_from_box(img, boxes=boxes)
    features = []
    for p in patches: 
        text_feature = img2text(
            p, 
            api_key=api_key, 
            query="are different pushpins divided by plastic wall?, or only contain one pushpin?"
        )["choices"][0]["message"]["content"]
        features.append(text_feature)
    print(features)
    assert features is not None