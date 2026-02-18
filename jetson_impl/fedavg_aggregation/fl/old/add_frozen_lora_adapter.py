import torch
from peft.tuners.lora import Linear as LoraLinear

def make_real_lora_linear(base_layer, lora_A_weight, lora_B_weight, r, lora_alpha, lora_dropout=0.05, trainable=False):
    device = next(base_layer.parameters()).device
    in_features = base_layer.in_features
    out_features = base_layer.out_features
    bias = base_layer.bias is not None

    lora_layer = LoraLinear(
        adapter_name="default",
        in_features=in_features,
        out_features=out_features,
        base_layer=base_layer,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=False,
        bias=bias,
        merge_weights=False
    )

    # Copy weights from received matrices
    with torch.no_grad():
        lora_layer.base_layer.weight.copy_(base_layer.weight.to(device))
        if base_layer.bias is not None:
            lora_layer.base_layer.bias.copy_(base_layer.bias.to(device))
        lora_layer.lora_A["default"].weight.copy_(lora_A_weight.to(device))
        lora_layer.lora_B["default"].weight.copy_(lora_B_weight.to(device))

    # Optionally freeze everything (for experts not trainable)
    if not trainable:
        for p in lora_layer.parameters():
            p.requires_grad = False

    return lora_layer

def attach_lora_to_expert(expert, gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B, r, lora_dropout=0.05, trainable=False):
    expert.gate_proj = make_real_lora_linear(expert.gate_proj, gate_proj_A, gate_proj_B, r, 2*r, lora_dropout, trainable)
    expert.up_proj = make_real_lora_linear(expert.up_proj, up_proj_A, up_proj_B, r, 2*r, lora_dropout, trainable)
    expert.down_proj = make_real_lora_linear(expert.down_proj, down_proj_A, down_proj_B, r, 2*r, lora_dropout, trainable)
    return expert



def find_lora_weights_received_params(received_params, layer, expert):
    base = f"base_model.model.model.layers.{layer}.mlp.experts.{expert}"

    gate_proj_A = received_params[f"{base}.gate_proj.lora_A.default.weight"]
    gate_proj_B = received_params[f"{base}.gate_proj.lora_B.default.weight"]

    up_proj_A = received_params[f"{base}.up_proj.lora_A.default.weight"]
    up_proj_B = received_params[f"{base}.up_proj.lora_B.default.weight"]

    down_proj_A = received_params[f"{base}.down_proj.lora_A.default.weight"]
    down_proj_B = received_params[f"{base}.down_proj.lora_B.default.weight"]

    return gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B



def add_or_update_lora_adapter(model, received_params, lora_params, trainable, rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(model)
    for layer_idx, layer in enumerate(model.base_model.model.model.layers):
        if layer_idx >= len(lora_params):
            break

        for expert_idx in lora_params[layer_idx]:
            gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B = [
                p.to(device) for p in find_lora_weights_received_params(received_params, layer_idx, expert_idx)
            ]

            if expert_idx not in trainable[layer_idx]:
                attach_lora_to_expert(layer.mlp.experts[expert_idx],
                                      gate_proj_A, gate_proj_B,
                                      up_proj_A, up_proj_B,
                                      down_proj_A, down_proj_B,
                                      rank, trainable=False)
            else:
                expert = layer.mlp.experts[expert_idx]
                with torch.no_grad():
                    expert.gate_proj.lora_A["default"].weight.copy_(gate_proj_A)
                    expert.gate_proj.lora_B["default"].weight.copy_(gate_proj_B)
                    expert.up_proj.lora_A["default"].weight.copy_(up_proj_A)
                    expert.up_proj.lora_B["default"].weight.copy_(up_proj_B)
                    expert.down_proj.lora_A["default"].weight.copy_(down_proj_A)
                    expert.down_proj.lora_B["default"].weight.copy_(down_proj_B)