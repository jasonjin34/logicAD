"""
Help functions for handling different backbone models
including followings:
1. ViT-B-16
2. ViT-B-16-plus-240
3. EVA02-B-16
"""
import re
import pdb
import torch
import argparse
from einops import rearrange
import open_clip

try:
    import eva_clip
except:
    print("EVA02-CLIP not installed")

import torch.nn.functional as F

def get_dino_encoders(backbone: str = "ViT-L-14"):
    if backbone == "ViT-B-16":
        model = torch.hub.load('facebookresearch/dino:main', "dino_vitb16")
    elif backbone == "ViT-L-14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')        
    else:
        raise ValueError("Backbone not supported, only support ViT-B-16 and ViT-L-14 for now.")
    return model.eval()

@torch.no_grad()
def get_dino_attn(x):
    vit = get_dino_encoders().cuda()
    B, nc, w, h = x.shape
    x = vit.patch_embed(x) 
    cls_tokens = vit.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + vit.interpolate_pos_encoding(x, w, h)
    for blk in vit.blocks[:-1]:
        x = blk(x)
    block = vit.blocks[-1] 
    x = block.norm1(x)
    B, N, C = x.shape
    attn = block.attn
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    output_attn_q = (q @ q.transpose(-2, -1)) * attn.scale * 1.5
    output_attn_k = (k @ k.transpose(-2, -1)) * attn.scale * 1.5
    output_attn = output_attn_q.softmax(dim=-1) + output_attn_k.softmax(dim=-1)
    return output_attn

def get_clip_encoders(
    backbone: str = 'ViT-B-16',
    pretrained: str = 'laion400m_e32'
):
    """
    extracted the pretrained clip model, tokenizer and hyperparameters setting
    """
    if backbone.startswith("EVA02-CLIP"):
       _clip = eva_clip
    else:
        _clip = open_clip
    try:
        if backbone.startswith("EVA02-CLIP"):
            model, _, preprocess = _clip.create_model_and_transforms(
                backbone, pretrained, force_custom_clip=True, device="cuda")
        else:
            model, _, preprocess = _clip.create_model_and_transforms(
                backbone, pretrained)
    except Exception as e:
        print(f"Error {e}, select the correct pretrained model")
    # change clip image encoder hyperparameters for few-shot learning
    model.visual.output_tokens = True
    model = model.eval()
    tokenizer = _clip.get_tokenizer(model_name=backbone)
    logit_scale = model.logit_scale
    clip_config = _clip.get_model_config(backbone)

    output_dict = {
        "clip_model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "logit_scale": logit_scale,
        "clip_config": clip_config
    }

    timm = check_timm_image_encoder(output_dict)
    output_dict["timm"] = timm
    if backbone.startswith("EVA02-CLIP"):
        output_dict["clipself_eva"] = True

    if timm:
        patch_size = [14, 16, 32]
        for patch in patch_size:
            if f"-{patch}"in backbone:
                clip_config["vision_cfg"]["patch_size"] = patch
                break
    output_dict["clip_config"] = clip_config
    return output_dict

def key_smoothing(x):
    atten = x @ x.transpose(1, -1)  # B L L
    atten = F.softmax(atten, dim=-1)
    x = atten @ x
    return x

def concat_csa_att(resblock, x, istimm=False):
    attn = resblock.attn
    num_heads = attn.num_heads
    _, B, E = x.size()
    head_dim = E // num_heads
    scale = head_dim ** -0.5
    if istimm:
        q = attn.q_proj(x)
        k = attn.q_proj(x)
        v = attn.q_proj(x)
    else:
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # [L, B, E]

    q = q.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
    q_attn = q @ q.transpose(1, 2) * scale  # [B * num_head, L, L]
    k_attn = k @ k.transpose(1, 2) * scale
    v_attn = v @ v.transpose(1, 2) * scale
    qk_attn = q @ k.transpose(1, 2) * scale
    attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1) + F.softmax(v_attn, dim=-1) + F.softmax(qk_attn, dim=-1)
    attn_output = attn_weights @ v
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, B, E)  # [L, B, E]

    if istimm:
        attn_output = attn.norm(attn_output)
        attn_output = attn.proj(attn_output)
        attn_output = attn.proj_drop(attn_output)
    else:
        attn_output = attn.out_proj(attn_output)
    return attn_output

def csa_attn(
    resblock, 
    x, 
    istimm=False, 
    isclipselfeva=False,
    attn_logit_scale = None,
    attn_dino=None,
):
    """
    TODO: clean this part of code, make it more modules 
    x input shape for timm model is [B, L, E]
    x input shape for non-timm model is [L, B, E]
    """
    attn = resblock.attn
    num_heads = attn.num_heads
    if istimm:
        B, N, C = x.shape
        head_dim = C // num_heads
        scale = head_dim ** -0.5
        q = attn.q_proj(x).reshape(B, N, num_heads, -1).transpose(1, 2)  # B, num_heads, N, head_dim
        k = attn.k_proj(x).reshape(B, N, num_heads, -1).transpose(1, 2)
        v = attn.v_proj(x).reshape(B, N, num_heads, -1).transpose(1, 2)

        if attn_logit_scale != -1:
            scale = attn_logit_scale

        q_attn = q @ q.transpose(-2, -1) * scale  # [B, num_head, L, L]
        k_attn = k @ k.transpose(-2, -1) * scale
        attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        x = attn_weights @ v  # [B, num_head, L, E // num_head]

        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn.norm(x)
        attn_output = attn.proj(x)
    elif isclipselfeva:
        '''
            cf https://github.com/wusize/CLIPSelf/blob/main/src/open_clip/eva_clip/eva_vit_model.py#L174
        '''
        B, N, C = x.shape
        q = F.linear(input=x, weight=attn.q_proj.weight, bias=attn.q_bias)
        k = F.linear(input=x, weight=attn.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=attn.v_proj.weight, bias=attn.v_bias)

        q = q.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, head_dim
        k = k.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, attn.num_heads, -1).permute(0, 2, 1, 3)

        if attn.rope:
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = attn.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = attn.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if attn_logit_scale != -1:
            scale = attn_logit_scale
        else:
            scale = attn.scale

        q_attn = q @ q.transpose(-2, -1) * scale  # [B, num_head, N, N]
        k_attn = k @ k.transpose(-2, -1) * scale
        attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)
        x = attn_weights @ v  # [B, num_head, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = attn.inner_attn_ln(x)
        x = attn.proj(x)
        attn_output = attn.proj_drop(x)
    else:
        L, B, E = x.size()
        head_dim = E // num_heads
        scale = head_dim ** -0.5
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # [L, B, E]
        q = q.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, B * num_heads, head_dim).transpose(0, 1)
        if attn_logit_scale != -1:
            scale = scale * attn_logit_scale
        q_attn = q @ q.transpose(1, 2) * scale  # [B * num_head, L, L]
        k_attn = k @ k.transpose(1, 2) * scale
        # qk_attn = q @ k.transpose(1, 2) * scale
        # attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1) + F.softmax(qk_attn, dim=-1)
        attn_weights = F.softmax(q_attn, dim=-1) + F.softmax(k_attn, dim=-1)

        if attn_dino is not None:
            attn_weights = rearrange(attn_dino, "B NH H W -> (B NH) H W")

        attn_output = attn_weights @ v  # [B * num_head, L, E // num_head]
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, B, E)  # [L, B, E]
        attn_output = attn.out_proj(attn_output)
    return attn_output

def check_timm_image_encoder(config):
    config_json = config["clip_config"]
    vision_cfg = config_json["vision_cfg"]
    for key in vision_cfg.keys():
        if "timm" in key:
            return True
    return False

def main(args):
    clip_init = get_clip_encoders(args.backbone, args.pretrained)
    model = clip_init["clip_model"]
    visual = model.visual
    if check_timm_image_encoder(clip_init):
        print("Timm image encoder")
    else:
        print("Not timm image encoder")

if __name__ == "__main__":
    # args = argparse.ArgumentParser()
    # args.add_argument("--backbone", type=str, default="EVA02-L-14")
    # args.add_argument("--pretrained", type=str, default="merged2b_s4b_b131k")
    # args = args.parse_args()
    # main(args)
    x = torch.randn(2, 3, 224, 224)
    vitb16 = get_dino_encoders()
    get_dino_attn(vitb16, x)