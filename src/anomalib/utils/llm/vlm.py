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

sys.path.append("/home/students/yuehan/projects/logicAD")

from pruning.attention_analysis import analyze_attention
from pruning.attention_analysis import analyze_attention, rank_attention_heads
from pruning.attention_analysis import build_attention_head_mask
from pruning.ffn_analysis import analyze_ffn
from pruning.ffn_analysis import analyze_ffn, build_ffn_neuron_mask
from pruning.structured_pruning import apply_attention_head_mask
from pruning.structured_pruning import apply_ffn_neuron_mask
from pruning.visualize import visualize_head_mask

LLAVA_PATH = "/home/students/yuehan/projects/logicAD/LLaVA"
sys.path.append(LLAVA_PATH)

from transformers import AutoTokenizer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

def load_llava_custom(model_path="liuhaotian/llava-v1.5-7b", model_name="llava-v1.5-7b", device="cuda"):
    offload_dir = "D:/huggingface_cache/offload"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device,
        torch_dtype=torch.float32,
        device_map="auto",
        offload_folder=offload_dir
    )
    return tokenizer, model, image_processor, context_len

LLAVA_ID = {
    "llava16": "liuhaotian/llava-v1.5-7b"  #"liuhaotian/llava-v1.5-13b"or"llava-hf/llava-v1.6-vicuna-13b-hf"
}

Text_SUM = [
    "facebook/bart-large-cnn"
]

def load_model(model_id, task, device):
    model_id = LLAVA_ID[model_id]
    pipe = pipeline(task, model=model_id, device=device)
    return pipe

def llava_inference(model, image_processor, tokenizer, prompt, path, max_new_tokens=300, temperature=0.2):
    # 加载图像
    if isinstance(path, str):
        image = Image.open(path).convert("RGB")
    else:
        path = path.permute(1, 2, 0).detach().numpy()
        image = Image.fromarray(path)

    # 构建对话 prompt
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # === Tokenization ===
    #tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
    inputs = tokenizer_image_token(prompt_text, tokenizer, return_tensors="pt").to(model.device)
    input_ids = inputs.unsqueeze(0).to(dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids).to(device=model.device)

    # === Image processing ===
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(dtype=model.dtype, device=model.device)

    # === [DEBUG] Check dtypes and shapes
    print("[debug] input_ids dtype:", input_ids.dtype)
    print("[debug] attention_mask dtype:", attention_mask.dtype)
    print("[debug] image_tensor dtype:", image_tensor.dtype)
    print("[debug] model dtype:", model.dtype)
    print("[debug] input_ids shape:", input_ids.shape)
    print("[debug] image_tensor shape:", image_tensor.shape)

    # === [PRUNING] Attention Head Pruning ===
    _, attn_maps, _, _ = analyze_attention(model, (input_ids, attention_mask, image_tensor))
    head_mask = build_attention_head_mask(attn_maps, top_k=800)

    print("head_mask is None?", head_mask is None)
    print("head_mask shape:", head_mask.shape if head_mask is not None else "None")
    print("heads to keep:", head_mask.sum().item() if head_mask is not None else "None", "/", head_mask.numel() if head_mask is not None else "None")

    print("Calling visualize_head_mask()...")
    #visualize_head_mask(head_mask, title="Top-800 Attention Head Mask")

    apply_attention_head_mask(model, head_mask)

    model.eval()
    torch.cuda.empty_cache()

    # 模型生成
    output = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=image_tensor,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens
    )

    # 解码输出
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text
