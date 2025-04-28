import torch.nn as nn
import torch

def apply_attention_head_mask(model, mask):
    """
    Replace each LLaMA layer's attention forward to control head outputs with the mask
    """
    assert mask is None or mask.shape[0] == len(model.model.layers), "Mask layer count mismatch"

    for layer_idx, layer in enumerate(model.model.layers):
        head_mask = None if mask is None else mask[layer_idx]

        def wrap_forward(orig_forward, head_mask=head_mask, layer_idx=layer_idx):
            def new_forward(self, hidden_states, *args, **kwargs):
                print(f"Executing wrapped attention forward of Layer {layer_idx}")

                print(f"[Layer {layer_idx}] Head mask active: {head_mask is not None}")
                attn_output, attn_weights, past_key_value = orig_forward(hidden_states, *args, **kwargs)

                #If no mask is used, return the original output directly

                if head_mask is None:
                    return attn_output, attn_weights, past_key_value

                #If mask is used, set the output of pruned heads to zero

                if attn_output.dim() == 3:
                    B, S, D = attn_output.shape
                    H = head_mask.shape[0]
                    head_dim = D // H
                    attn_output = attn_output.view(B, S, H, head_dim)
                    head_mask_float = head_mask.to(attn_output.device).to(attn_output.dtype).view(1, 1, H, 1)
                    attn_output = attn_output * head_mask_float
                    attn_output = attn_output.view(B, S, D)

                    #apply or not
                    if head_mask is not None:
                        with torch.no_grad():
                            zeroed_heads = (~head_mask).nonzero(as_tuple=False).tolist()
                            if zeroed_heads:
                                B, S, D = attn_output.shape
                                H = head_mask.shape[0]
                                head_dim = D // H
                                attn_output_heads = attn_output.view(B, S, H, head_dim)
                                masked_out = attn_output_heads[:, :, ~head_mask, :]
                                zero_check = torch.all(masked_out == 0)
                                print(f"Pruned heads zeroed: {zero_check.item()} | Zeroed heads: {len(zeroed_heads)}")

                return attn_output, attn_weights, past_key_value

            return new_forward.__get__(layer.self_attn, layer.self_attn.__class__)

        layer.self_attn.forward = wrap_forward(layer.self_attn.forward)

    print("Attention head mask applied (head_mask=None means full attention).")

def apply_ffn_neuron_mask(model, mask_dict):
    """
    Apply the FFN neuron mask to each layer of the LLaMA model.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        ffn_mask = mask_dict.get(layer_idx, None)

        def wrap_ffn_forward(orig_forward, ffn_mask=ffn_mask, layer_idx=layer_idx):
            def new_forward(self, x):
                out = orig_forward(x)  #original output

                if ffn_mask is None:
                    return out  #no pruning

                # out: (B, S, D), mask: (D,)
                mask = ffn_mask.to(out.device, dtype=out.dtype).view(1, 1, -1)
                out = out * mask

                print("model dtype example:", model.model.layers[0].mlp.up_proj.weight.dtype)
                print("mask dtype:", ffn_mask.dtype)
                print("x dtype:", out.dtype)

                return out

            return new_forward.__get__(layer.mlp, layer.mlp.__class__)

        layer.mlp.forward = wrap_ffn_forward(layer.mlp.forward)

    print("FFN neuron mask applied to model.")


