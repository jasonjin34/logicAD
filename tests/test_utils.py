"""Test for the utils functions"""

from PIL import Image
import numpy as np
import torch

from anomalib.utils.llm.base import (
    img2text,
    txt2sum,
    cos_sim,
    txt2embedding,
)

API = "./keys/gpt.json"


def test_img2text():
    global API
    image_path = "./tests/test_data/few_pushpin.png"
    output = img2text(image_path, API, model="gpt-4o", query="How many pushpins are there?")
    print(output)
    assert output is not None


#
#
def test_txt2sum():
    global API
    good_img = "./datasets/MVTec_Loco/original/breakfast_box/test/good/007.png"
    query = "what is on the left side of image? and what is on the right side of image?"
    train = img2text(good_img, API, query=query)["choices"][0]["message"]["content"]
    train = txt2sum(input_text=train, few_shot_message="number objects, ...", api_key=API)
    print(train)
    assert train is not None


#
def test_txt2embedding():
    global API
    good_img = "./datasets/MVTec_Loco/original/breakfast_box/test/good/007.png"
    query = "what is on the left side of image? and what is on the right side of image?"
    train = img2text(good_img, API, query=query)["choices"][0]["message"]["content"]
    train = txt2sum(input_text=train, few_shot_message="number objects, ...", api_key=API)
    train_embedding = txt2embedding(input_text=train, api_key=API)
    assert train_embedding is not None


#
def test_cos_sim():
    a = [0.1, 0.2, 0.3]
    b = [0.11, 0.21, 0.31]
    assert cos_sim(a, b) > 0.9


def test_encode_image():
    global API
    image_path = "./tests/test_data/few_pushpin.png"
    # test for image path
    path_results0 = img2text(image_path, API, model="gpt-4o", query="How many pushpins are there?")
    image = np.asarray(Image.open(image_path))
    path_results1 = img2text(image, API, model="gpt-4o", query="How many pushpins are there?")
    # convert numpy to tensor
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    path_results2 = img2text(image, API, model="gpt-4o", query="How many pushpins are there?")
    assert None not in [path_results0, path_results1, path_results2]
