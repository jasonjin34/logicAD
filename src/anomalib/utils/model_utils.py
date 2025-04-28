import torch
import os

os.environ["HF_HOME"] = "D:/huggingface_cache"

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import requests
from transformers import pipeline
from PIL import Image
import time
import glob
import sys

LLAVA_PATH = "D:/MA/LLaVA"
sys.path.append(LLAVA_PATH)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.conversation import conv_templates

from llava.mm_utils import tokenizer_image_token

def load_llava_custom(model_path="liuhaotian/llava-v1.5-13b", model_name="llava-v1.5-13b", device="cuda"):
    offload_dir = "D:/huggingface_cache/offload"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder= offload_dir  
    )
    return tokenizer, model, image_processor, context_len

LLAVA_ID = {
    "llava16": "llava-hf/llava-v1.6-vicuna-13b-hf"
}

Text_SUM = [
    "facebook/bart-large-cnn"
]

def test_img2text_extraction():
    dir_path = "/home/erjin/git/Logic_AD_LLM/datasets/MVTec_Loco/original/breakfast_box/test/logical_anomalies/" 
    path_list = glob.glob(os.path.join(dir_path, "*.png"))

    pip = load_model_pipeline(LLAVA_ID[0], "image-to-text", "cuda:1")
    prompt = "USER: <image>\n Can you please describe this image? Does the image contain scratches?\nASSISTANT:"
    start_time = time.time()

    for p in path_list:
        print(p)
        text = llava_inference_pipeline(pip, prompt, p)
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
    pipe = load_model_pipeline(Text_SUM[0], task, "cuda:1")
    text = text + "summarize the text as json format {location: object: number}"
    print(pipe(text))

def test_raw_inference():
    model_path = "liuhaotian/llava-v1.5-13b"  
    tokenizer, model, image_processor, context_len = load_llava_custom(model_path=model_path)

    # Force load vision_tower in case it is None
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if vision_tower is not None and not getattr(vision_tower, 'is_loaded', True):
            vision_tower.load_model()
            vision_tower.to(device=model.device, dtype=torch.float16)


    image_path = "D:/MA/LogicAD/datasets/juice_bottle/test/logical_anomalies/000.png"
    image = Image.open(image_path).convert("RGB")
    prompt = "Please describe the location and number of objects in the image."

    # construct prompt
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(dtype=torch.float16, device=model.device)
    print("dtype:", image_tensor.dtype, "device:", image_tensor.device)

    """ inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    print("input_ids:", input_ids)
    print("decode input_ids:", tokenizer.batch_decode(input_ids))
    attention_mask = inputs["attention_mask"].to(model.device) """

    inputs = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").to(model.device)
    input_ids = inputs.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids).to(model.device) 

    print("input_ids shape:", input_ids.shape)
    print("image_tensor shape:", image_tensor.shape)

    output = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,  
        do_sample=True,
        temperature=0.2,
        max_new_tokens=50
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("LLaVA description output:", output_text)