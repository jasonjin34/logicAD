"""help function for notebook demo"""
import os, math

import ipdb
import open_clip
import torch
from PIL import Image
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import umap
import matplotlib.pyplot as plt
from text_classifier import build_text_classifier


def get_clip_encoders(
    backbone: str = 'ViT-B-16',
    pretrained: str = 'laion400m_e32'
):
    """
    extracted the pretrained clip model, tokenizer and hyperparameters setting
    """
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained)
    except Exception as e:
        print(f"Error {e}, select the correct pretrained model")

    # change clip image encoder hyperparameters for few-shot learning
    model.visual.output_tokens = True
    model = model.eval()

    visual_encoder = model.visual.eval()
    tokenizer = open_clip.get_tokenizer(model_name=backbone)

    logit_scale = model.logit_scale
    clip_config = open_clip.get_model_config(backbone)

    return {
        "clip_model": model,
        "preprocess": preprocess,
        "visual_encoder": visual_encoder,
        "tokenizer": tokenizer,
        "logit_scale": logit_scale,
        "clip_config": clip_config
    }


def img_to_tensor(img_path:str, config_init: dict):
    preprocess = config_init["preprocess"]
    with Image.open(img_path) as img:
        return preprocess(img)


def csa_attn(resblock, x):
    attn = resblock.attn
    num_heads = attn.num_heads
    _, B, E = x.size()
    head_dim = E // num_heads
    scale = head_dim ** -0.5
    qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)  # [L, B, E]
    q = q.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    q_attn = q @ q.transpose(1, 2) * scale  # [B * num_head, L, L]
    k_attn = k @ k.transpose(1, 2) * scale
    attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
    attn_output = attn_weights @ v  # [B * num_head, L, E // num_head]
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, B, E)  # [L, B, E]
    attn_output = attn.out_proj(attn_output)
    return attn_output


def image_encoder(x,  config_init: dict = None, csa: bool = True):
    if len(x.shape) != 4:
        x = x[None, ...]
    visual_encoder = config_init["visual_encoder"]
    vit = visual_encoder
    x = vit.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = torch.cat(
        [vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
         x], dim=1)  # shape = [*, grid ** 2 + 1, width]

    x = x + vit.positional_embedding.to(x.dtype)
    x = vit.patch_dropout(x)
    x = vit.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND

    resblocks = vit.transformer.resblocks

    for block in resblocks[:11]:
        x = block(x)
    block = resblocks[-1]
    if csa:
        x = x + csa_attn(block, block.ln_1(x))
        x = x + block.mlp(block.ln_2(x))
    else:
        x = block(x)

    x = x.permute(1, 0, 2)
    x = vit.ln_post(x) @ vit.proj
    x = F.normalize(x, dim=-1, p=2)

    return {
        "cls": x[:, 0, :],
        "tokens": x[:, 1:, :]
    }


def img_to_numpy(path, idx, patch_size=16, resize_window = 224):
    with Image.open(path) as img:
        img = np.asarray(img)
        if img.shape[-1] == 3:
            width = img.shape[1]
        else:
            width = img.shape[-1]
        scaled_patch_size = int(patch_size * width / resize_window)
        x, y = divmod(idx, 14)
        y = y * scaled_patch_size
        x = x * scaled_patch_size
        img = cv.rectangle(img, (y, x), (y + scaled_patch_size, x + scaled_patch_size), (0, 255, 0), 5)
        return img


def feature_extraction_dino(
        img,
        backbone_size: str = "large",
):
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    model = backbone_model
    if len(img.shape) == 3:
        img = img[None, ...]
    img = model.patch_embed(img)
    for b in model.blocks:
        img = b(img)
    img = img.detach().numpy()
    return img


def umap_plot(
    img_path,
    config_init,
    n_neighbors: int = 2,
    min_dist: float = 0.1,
    n_components: int = 2,
    idx_plot_module: int = 2,
    backbone: str = "clip"
):
    img = img_to_tensor(img_path, config_init)
    if backbone == "clip":
        patch_tokens = image_encoder(img, config_init)["tokens"]
        embeddings = patch_tokens.detach().numpy()[0]
    elif backbone == "dino":
        embeddings = feature_extraction_dino(img)[0]

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components
    )
    u = fit.fit_transform(embeddings)
    color = np.arange(len(embeddings))

    figure, axis = plt.subplots(1, 2, figsize=(15*2, 15), width_ratios=(3, 1))
    axis[0].scatter(u[:, 1], u[:, 0], c=color)
    for idx, point in enumerate(u):
        if idx % idx_plot_module == 0:
            axis[0].annotate(idx, (point[1], point[0]))
    axis[0].set_title("umap image feature embedding", fontsize=20)
    if backbone == "dino":
        patch_size = 14
    img = plot_img_with_idx_annotation(patch_size=patch_size, path=img_path)
    axis[1].imshow(img, cmap="gray")
    axis[1].set_title("query image with patch index", fontsize=20)
    plt.tight_layout()
    plt.show()


def plot_img_with_idx_annotation(path, patch_size=16, resize_window=224):
    with Image.open(path) as img:
        img = np.asarray(img)

        if img.shape[-1] == 3:
            width = img.shape[1]
        else:
            width = img.shape[-1]

        scaled_patch_size = int(patch_size * width / resize_window)
        sequence_length = (resize_window // patch_size) ** 2
        for idx in range(sequence_length):
            y, x = divmod(idx, resize_window // patch_size)
            y = y * scaled_patch_size
            x = x * scaled_patch_size
            img = cv.putText(img, str(idx), (x + scaled_patch_size // 3, y + scaled_patch_size // 3),
                                 cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    return img


def generate_heatmap(
    image_features,
    text_features,
    config_init
):
    """
    generate foreground and background heatmap for anomaly detection
    Args:
        image_features:  image_feature has shape [B, L, D]
        text_features: text feaures for text embedding, could be empty prompt, category prompt or background prompt
                        [C, D]
    Returns:
        heatmap
    """
    image_features = image_encoder(image_features, config_init)["tokens"]
    heatmap = image_features @ text_features.T  # shape [B, L, C]
    heatmap = heatmap.softmax(dim=1)  # shape [B, L, C]
    heatmap = heatmap.transpose(-1, 1)
    return heatmap



def plot_attention(
    query_img_path: str,
    query_patch_idx: int,
    config_init: dict,
    #ref_img_path: str = None,
    img_size: int = 224,
    fig_size: int = 20
):
    def post_processing_heatmap(heatmap):
        width_heatmap = int(heatmap.shape[-1] ** 0.5)
        heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1], width_heatmap, width_heatmap)
        heatmap = F.interpolate(
            heatmap,
            (img_size, img_size),
            mode="bilinear")[0, ...].detach().numpy()
        return heatmap

    # build text embedding for all categories plus an empty category
    categories = ["background", "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"] 
    category = query_img_path.split(os.sep)[-4]
    text_embedding = build_text_classifier(config_init)
    for c in categories:
        text_embedding = torch.cat([
            text_embedding,
            build_text_classifier(config_init, c, background=False)
        ], dim=0)

    query_img_ori = img_to_tensor(query_img_path, config_init)
    query_img = image_encoder(query_img_ori, config_init, csa=False)["tokens"]
    #if ref_img_path is not None:
    #    ref_img = img_to_tensor(ref_img_path, config_init)
    #    ref_img = image_encoder(ref_img, config_init)["tokens"]
    #else:
    #    ref_img = query_img.clone()

    # shape [C, H, W]
    text_heatmaps = generate_heatmap(query_img_ori, text_embedding, config_init)
    text_heatmaps = post_processing_heatmap(text_heatmaps)

    #atten_map = query_img @ ref_img.transpose(-1, 1)
    #heatmap = atten_map[:, query_patch_idx:query_patch_idx+1, :]
    #most_similar_patch_idx = heatmap.max(dim=-1)[1][0][0].detach().numpy()
    #heatmap = post_processing_heatmap(heatmap)

    #generate self attention map
    #self_attention_map = query_img @ query_img.transpose(-1, 1)
    #self_attention_map = self_attention_map[:, query_patch_idx:query_patch_idx+1, :]
    #self_attention_map = post_processing_heatmap(self_attention_map).squeeze(0)

    titles = [
        "query image (test)",
        f"category {category} (ground truth) heatmap",
        "empty prompt heatmap",
        "background heatmap",
        #"self attention heatmap"
    ] + [f"category {c} heatmap" for c in categories[1:] if c != category]

    figure, axis = plt.subplots(3, math.ceil(len(titles)//3), figsize=(fig_size, fig_size*0.5))
    images = [
        img_to_numpy(query_img_path, query_patch_idx),
        #img_to_numpy(ref_img_path, most_similar_patch_idx),
        #heatmap,
        #self_attention_map,
        text_heatmaps[categories.index(category)+1],
        text_heatmaps[0],
        text_heatmaps[1]
    ] + [text_heatmaps[i+1] for i in range(len(categories)) if i != categories.index(category)]

    for i in range(len(titles)):
        x, y = divmod(i, 3)
        if i == 0:
            axis[y,x].imshow(images[i], cmap="gray")
        else:
            axis[y,x].imshow(images[i])
        axis[y,x].set_title(titles[i], fontsize=10)
        axis[y,x].axis('off')
    plt.show()


if __name__ == "__main__":
    config_init = get_clip_encoders()
    good_img_path = "./datasets/MVTec/screw/train/good/002.png"
    query_img_path = "./datasets/MVTec/screw/test/scratch_neck/011.png"
    plot_attention(query_img_path, 86, config_init, good_img_path)

