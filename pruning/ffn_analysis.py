import torch
import torch.nn.functional as F

def analyze_ffn(model, input_ids, attention_mask, image_tensor):
    print("Running FFN neuron activity analysis...")
    ffn_activations = {}
    hooks = []

    def save_ffn_output(name):
        def hook_fn(module, input, output):
            # input[0]: (batch, seq, dim), output: (batch, seq, dim)
            act = output.detach().cpu().mean(dim=(0, 1))  # mean over batch and seq
            ffn_activations[name] = act
        return hook_fn

    for name, module in model.named_modules():
        if hasattr(module, 'mlp'):
            h = module.mlp.register_forward_hook(save_ffn_output(name))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            images=image_tensor,  
            do_sample=True,
            max_new_tokens=200
        )

    for h in hooks:
        h.remove()

    for name, act in ffn_activations.items():
        print(f"Layer {name}: FFN mean activations {act.shape}\n{act}")

    print("FFN neuron activity analysis complete.")
    return ffn_activations

def build_ffn_neuron_mask(ffn_activations, keep_ratio=0.5):
    """
    Build pruning mask based on average activation of each FFN layer
    """
    ffn_mask_dict = {}
    for name, act in ffn_activations.items():
        #Extract layer index
        try:
            layer_idx = int(name.split("layers.")[1].split(".")[0])
        except Exception as e:
            print(f"[warn] Skipping layer name parsing: {name}")
            continue

        #Compute threshold for keeping neurons
        num_neurons = act.numel()
        k = int(num_neurons * keep_ratio)
        topk = torch.topk(act.abs().to(torch.float32), k)  #Keep top-k neurons by absolute activation values

        threshold = topk.values.min()

        #build mask
        mask = (act.abs() >= threshold)
        ffn_mask_dict[layer_idx] = mask

        print(f"[mask] Layer {layer_idx}: kept {mask.sum().item()}/{num_neurons} neurons")

    return ffn_mask_dict
