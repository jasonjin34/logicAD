import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import requests
from transformers import pipeline
from PIL import Image
import time
import glob
import os

LLAVA_ID = {
    "llava16": "llava-hf/llava-v1.6-vicuna-13b-hf"
}

Text_SUM = [
    "facebook/bart-large-cnn"
]

def load_model(model_id, task, device):
    model_id = LLAVA_ID[model_id]
    pipe = pipeline(task, model=model_id, device=device)
    return pipe


def llava_inference(model, prompt, path, max_new_tokens=300):
    if isinstance(path, str):
        image = Image.open(path)
    else:
        path = path.permute(1, 2, 0).detach().numpy()
        image = Image.fromarray(path)

    prompt = f"USER: <image>\n {prompt} \nASSISTANT:"
    outputs = model(
        image, prompt=prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
    text = outputs[0]["generated_text"]
    return text


def test_img2text_extraction():
    dir_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/breakfast_box/test/logical_anomalies/" 
    path_list = glob.glob(os.path.join(dir_path, "*.png"))

    pip = load_model(LLAVA_ID[0], "image-to-text", "cuda:1")
    prompt = "USER: <image>\n Can you please describe this image? Does the image contain scratches?\nASSISTANT:"
    start_time = time.time()

    for p in path_list:
        print(p)
        text = llava_inference(pip, prompt, p)
        print(text)
        print("=====================================")
    
    total_time = time.time() - start_time
    print(f"Time taken: {total_time}, average inference per image {total_time/len(path_list)}",)


def txt2txt_inference(text, model):
    outputs = model(text)
    return outputs

def text_sum():
    text = "In the image, the left side of the container holds two oranges and one nectarine or peach. The right side of the container contains granola, sliced bananas, and almonds."
    task = "summarization"
    pipe = load_model(Text_SUM[0], task, "cuda:1")
    text = text + "summarize the text as json format {location: object: number}"
    print(pipe(text))


if __name__ == "__main__":
    text_sum()