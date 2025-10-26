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
import pdb
from openai._types import NotGiven
from openai import AzureOpenAI

from .vlm import llava_inference


def image2base64(image: Union[np.ndarray, Image.Image], image_format: str = "JPEG", quality: int = 100) -> str:
    try:
        if isinstance(image, np.ndarray):
            im = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            im = image
        else:
            raise ValueError(f"image should be np.ndarray or PIL.Image, not {type(image)}")
    except Exception as e:
        print(f"Error in converting image to base64")

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


def key_extraction(file_path, key_name="gpt"):
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"GPT api key file not found: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
        key = data[key_name]
    return key


def resize_image(image, img_size=None):
    img = Image.open(image)
    if img_size is None:
        return img
    else:
        if not isinstance(img_size, int):
            img_size = img_size[0]
        wpercent = img_size / float(img.size[0])
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((img_size, hsize), Image.Resampling.LANCZOS)
        return img


# Function to encode the image
def encode_image(image, image_format="png", img_size=None):
    """
    convert the code to base64
    """
    if isinstance(image, str):  # if the image is a path
        image_path = image
        img = resize_image(image_path, img_size=img_size)
        img = image2base64(img, image_format=image_format)
        # img = base64.b64encode(img).decode("utf-8")
    else:  # if the image is a numpy array or tensor
        if isinstance(image, torch.Tensor):
            # convert B, C, H, W to  H, W, C as numpy format
            image = image.squeeze(0).permute(1, 2, 0).numpy()
        img = image2base64(image, image_format=image_format)
    return img


def img2text(
    image,
    api_key,
    model_name="gpt-4o-az",  # gpt-4o, gpt-4o-az, llava
    model=None,
    query="How many pushpins are there?",
    temperature=0.1,
    top_p=0.1,
    img_size=128,
    max_tokens=300,
    seed=42,
    ref_img=None,
):
    """
    image text extracton using LLM model, so far only support gpt-4o model
    # add other LLM model, particular for some small model for the ablation study
    args:
        image: str, path to the image, numpy array or tensor
    """
    if model_name in ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-az"]:
        if model_name == "gpt-4o-az":
            api_dict = key_extraction(api_key, key_name="gpt-az")
            api_key = api_dict["key"]
            endpoint = api_dict["endpoint"]  # specific end point for GPT4V
        else:
            api_key = key_extraction(api_key)
            endpoint = "https://api.openai.com/v1/chat/completions"
        # Getting the base64 string
        base64_image = encode_image(image, img_size=img_size)

        image_query = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

        if isinstance(query, list):
            list_text_query_dict = [{"type": "text", "text": q} for q in query]
        else:
            list_text_query_dict = [{"type": "text", "text": query}]
        user_content = [image_query] + list_text_query_dict

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an AI assistant that helps people find information."}
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if model_name == "gpt-4o":
            payload["model"] = "gpt-4o"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        else:
            headers = {"Content-Type": "application/json", "api-key": api_key}

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            output = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Have problem in extracting text, Error in the request: {e}")
            output = None
    else:
        output = llava_inference(model=model, prompt=query, path=image, max_new_tokens=max_tokens)

    return output


def load_azure_openai_client(api_key, model_name_key="gpt-4o"):
    api_dict = key_extraction(api_key, key_name="gpt-az")
    client = AzureOpenAI(
        api_key=api_dict["key"], api_version=api_dict["api_version"], azure_endpoint=api_dict["azure_endpoint"]
    )
    return {"client": client, "model_name": api_dict[model_name_key]}


def txt2embedding(
    input_text="",
    api_key=None,
    model="text-embedding-3-large",
):
    """
    use openai to get the embedding of the text
    """
    assert model in ["text-embedding-3-large", "text-embedding-3-large-az"], "model not supported"

    if model == "text-embedding-3-large-az":
        output = load_azure_openai_client(api_key, model_name_key="embedding")
        client, model = output["client"], output["model_name"]
    else:
        api_key = key_extraction(api_key)
        client = OpenAI(api_key=key_extraction(api_key))

    response = client.embeddings.create(input=input_text, model=model)
    return response.data[0].embedding


def txt2sum(
    input_text="",
    few_shot_message="",
    api_key=None,
    model="gpt-4o",
    system_message="You are a helpful assistant designed to output JSON.",
    seed=42,
    max_token=200,
    top_p=None,
    temp=None,
):
    """
    use openai to get the summarization of the text
    """

    if top_p is None:
        top_p = NotGiven
    if temp is None:
        temp = NotGiven

    if model == "gpt-4o-az":
        output = load_azure_openai_client(api_key)
        client, model = output["client"], output["model_name"]
    elif model == "gpt-4o":
        api_key = key_extraction(api_key)
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError("model not supported")

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
        # seed=seed,
    )

    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output


def txt2formal(
    api_key,
    model="gpt-4o",
    max_token=100,
    **prompt,
):
    """
    use openai to get the formalization of the text
    """

    if model == "gpt-4o-az":
        output = load_azure_openai_client(api_key)
        client, model = output["client"], output["model_name"]
    elif model == "gpt-4o":
        api_key = key_extraction(api_key)
        client = OpenAI(api_key=api_key)
    else:
        raise ValueError("model not supported")

    prompt0 = prompt["prompt"]
    syn_rules = prompt["syn_rules"]

    if prompt.get("k_shot", True):
        k_shot = prompt["k_shot"]
    else:
        k_shot = ""

    if prompt.get("query", True):
        query = prompt["query"]
    else:
        query = ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt0 + syn_rules + k_shot + query,
            }
        ],
        max_tokens=max_token,
    )

    try:
        output = response.choices[0].message.content
    except Exception as e:
        output = None
    return output
