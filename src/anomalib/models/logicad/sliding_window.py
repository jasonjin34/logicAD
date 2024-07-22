# This the help function for using GroundingDINO model as preprocessing step for the LLM model

from typing import Tuple
import torch
import numpy as np
from PIL import Image

from groundingdino.util.inference import predict
import groundingdino.datasets.transforms as T


def dino_image_transform(image) -> Tuple[np.array, torch.Tensor]:
    """
    modify from https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py
    """
    if isinstance(image, list):
        image = image[0]
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, str):
        image_source = Image.open(image).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
    else:
        raise ValueError("Input should be a path to an image file.")
    return image, image_transformed

def patch_extraction_from_box(image, boxes, patch=1):
    """
    the output should be a list of patches
    """

    # assume the image would be tensor, need to change to numpy if necessary
    if isinstance(image, np.ndarray):
        h, w, _ = image.shape # height, and width 
        image = torch.tensor(image).permute(2, 0, 1)
    else:
        h, w = image.shape[-2:]
        print(w, h)

    if w > h:
        patch = int(h // 3 * patch)
    else:
        patch = int(w // 3 * patch)

    output_image = []
    for box in boxes:
        # convert box to image dim
        # box format, center w, center h
        cw, ch = int(box[0] * w), int(box[1] * h)
        sw = cw - patch // 2
        ew = cw + patch // 2
        sh = ch - patch // 2
        eh = ch + patch // 2
        try:
            if sw <= 0:
                sw = 0
            if sh <= 0:
                sh = 0
            if ew >= w:
                ew = -1
            if eh >= h:
                eh = -1

        except Exception as e:
            import pdb; pdb.set_trace()
        output_image.append(image[:, sh:eh, sw:ew])
    return output_image 


QUERY_DICT = {
    "splicing_connectors": "connectors without cable",
    "pushpins": "pushpin"
}
    

def get_bbx_from_query(
    image, 
    model, 
    query="pushpins", 
    box_threshold=0.35, 
    text_threshold=0.35,
    device="cuda"
):
    global QUERY_DICT
    if query in QUERY_DICT:
        query = QUERY_DICT[query]
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=query, 
        box_threshold=box_threshold, 
        text_threshold=text_threshold,
        device=device
    )
    return boxes, logits, phrases


def convert_centroid_to_numpy(boxes):
    if not isinstance(boxes, np.ndarray):
        boxes = boxes.detach().cpu().numpy()
    output_list: list = []
    for b in boxes:
        x, y, w, h = b
        output_list.append([x, y])
    return output_list


def calculate_centroidmap_disance(v1, v2):
    def l2_dis(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    dis_list = []
    for p1 in v1:
        min_dis = 100
        for p2 in v2:
            temp_dis = l2_dis(p1, p2)
            if temp_dis < min_dis:
                min_dis = temp_dis
        dis_list.append(min_dis)
    return dis_list