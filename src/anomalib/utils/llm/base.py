# this is the utils function for the LLM based model and algorithm
import base64
import requests
import json
import os
from openai import OpenAI
import torch

from numpy import dot
import numpy as np
from numpy.linalg import norm
from typing import Union
from PIL import Image
import io

def image2base64(image: Union[np.ndarray, Image.Image], image_format: str = "JPEG", quality: int = 100) -> str:
    if isinstance(image, np.ndarray):
        im = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        im = image
    else:
        raise ValueError(f"image should be np.ndarray or PIL.Image, not {type(image)}")

    buffered = io.BytesIO()
    im.save(buffered, format=image_format, quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def cos_sim(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    cos_sim = dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def key_extraction(file_path):
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"GPT api key file not found: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        key = data["gpt"]
    return key


# Function to encode the image
def encode_image(image, image_format="png"):
    """
    convert the code to base64
    """
    if isinstance(image, str): # if the image is a path
        image_path = image
        with open(image_path, "rb") as image_file:
            img = image_file.read()
            img = base64.b64encode(img).decode("utf-8")
    else: # if the image is a numpy array or tensor
        if isinstance(image, torch.Tensor):
            # convert B, C, H, W to  H, W, C as numpy format
            image = image.squeeze(0).permute(1, 2, 0).numpy()
        img = image2base64(image, image_format=image_format)
    return img


def img2text(
    image,
    api_key,
    model="gpt-4o",
    query="How many pushpins are there?",
):
    """
    image text extracton using LLM model, so far only support gpt-4o model
    # TODO
    # add other LLM model, particular for some small model for the ablation study

    args:
        image: str, path to the image, numpy array or tensor
    """

    api_key = key_extraction(api_key)
    # Getting the base64 string
    base64_image = encode_image(image)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{query}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 300,
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        output = response.json()
    except Exception as e:
        print(f"Have problem in extracting text, Error in the request: {e}")
        output = None
    return output


def txt2embedding(
    input_text="",
    api_key=None,
    model="text-embedding-3-large",
):
    """
    use openai to get the embedding of the text
    """
    client = OpenAI(api_key=key_extraction(api_key))

    response = client.embeddings.create(input=input_text, model=model)
    return response.data[0].embedding


def txt2sum(
    input_text="",
    few_shot_message="",
    api_key=None,
    model="gpt-4o",
    system_message="You are a helpful assistant designed to output JSON.",
    max_token=100,
):
    """
    use openai to get the summarization of the text
    """
    api_key = key_extraction(api_key)
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": f"Can you summarize as following format, {few_shot_message}? {input_text}"},
        ],
    )
    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output

