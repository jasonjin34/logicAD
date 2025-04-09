import torch

def analyze_attention(model, inputs):
    input_ids, attention_mask, image_tensor = inputs

    attention_scores = {}
    hooks = []

    def hook_fn(layer_name):
        def fn(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if isinstance(attn, torch.Tensor) and attn.dim() == 4:
                    print(f"[hook] Captured {layer_name} attention: shape = {attn.shape}")
                    attention_scores[layer_name] = attn.detach().cpu()
                else:
                    print(f"[hook] {layer_name} returned invalid tensor or shape")
            else:
                print(f"[hook] {layer_name} output is not a tuple or lacks attention")
        return fn

    try:
        for idx, layer in enumerate(model.model.layers):
            name = f"language_layer_{idx}.self_attn"
            hooks.append(layer.self_attn.register_forward_hook(hook_fn(name)))
            print(f"[register] Hooking {name}")
    except AttributeError as e:
        print(f"language hook error {e}")
        return {}, [], 0, 0

    model.eval()
    with torch.no_grad():
        print(f"[debug] input_ids shape: {input_ids.shape}")
        print(f"[debug] image_tensor shape: {image_tensor.shape}")
        model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,
            output_attentions=True,
            max_new_tokens=1
        )

    for h in hooks:
        h.remove()

    attn_stats = {}
    for name, tensor in attention_scores.items():
        if tensor.dim() >= 4:
            var = torch.var(tensor, dim=-1).mean().item()
            attn_stats[name] = {
                'variance': var,
                'shape': tuple(tensor.shape)
            }
        else:
            print(f"[warn] Tensor from {name} is invalid for stats")

    attention_maps = []
    for idx in range(len(model.model.layers)):
        key = f"language_layer_{idx}.self_attn"
        if key in attention_scores:
            attention_maps.append(attention_scores[key])
        else:
            print(f"[warn] No attention map for {key}, filling None")
            attention_maps.append(None)

    total_token_len = attention_maps[0].shape[-1] if attention_maps[0] is not None else 0
    vision_token_len = total_token_len - input_ids.shape[1]

    print("Language attention analysis complete.")
    return attn_stats, attention_maps, vision_token_len, total_token_len

def rank_attention_heads(attention_maps, top_k=10):
    """
    # Calculate variance per attention head and select top-k by importance

        top_heads: List[Dict]：
            {
                "layer": int,
                "head": int,
                "variance": float
            }
    """
    ranked_heads = []
    for layer_idx, attn_tensor in enumerate(attention_maps):
        # attn_tensor: (B, H, S, S)
        var = torch.var(attn_tensor, dim=(-1, -2))  # (B, H)
        var = var.mean(dim=0)  # mean over batch -> (H,)
        for head_idx, head_var in enumerate(var):
            ranked_heads.append({
                "layer": layer_idx,
                "head": head_idx,
                "variance": head_var.item()
            })

    # rank and give top-k
    top_heads = sorted(ranked_heads, key=lambda x: x["variance"], reverse=True)[:top_k]
    return top_heads

def build_attention_head_mask(attention_maps, top_k=None, threshold=None):
    """
    Build pruning mask for attention heads
    """
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1] if attention_maps[0] is not None else 0
    head_variances = []

    for layer_idx, attn in enumerate(attention_maps):
        if attn is None:
            continue
        var = torch.var(attn, dim=(-1, -2))  # shape: (B, H)
        var = var.mean(dim=0)  # shape: (H,)
        for head_idx, v in enumerate(var):
            head_variances.append({
                "layer": layer_idx,
                "head": head_idx,
                "variance": v.item()
            })

    if top_k is not None:
        selected = sorted(head_variances, key=lambda x: x["variance"], reverse=True)[:top_k]
        selected_set = set((h["layer"], h["head"]) for h in selected)
    elif threshold is not None:
        selected_set = set((h["layer"], h["head"]) for h in head_variances if h["variance"] >= threshold)
    else:
        raise ValueError("Must specify top_k or threshold.")

    # Build the mask: True = keep, False = prune
    mask = torch.zeros((num_layers, num_heads), dtype=torch.bool)
    for l in range(num_layers):
        for h in range(num_heads):
            if (l, h) in selected_set:
                mask[l, h] = True

    return mask
