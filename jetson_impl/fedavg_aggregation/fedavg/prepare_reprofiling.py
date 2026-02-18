import torch
from peft.tuners.lora import Linear as LoraLinear, LoraLayer
from peft import LoraConfig
from bitsandbytes.nn import Linear4bit
import math

def add_lora(qlinear, rank=4, alpha=8, dropout=0.05):
    # 1. Get layer properties directly from the quantized layer
    in_features = qlinear.in_features
    out_features = qlinear.out_features
    bias = qlinear.bias is not None
    
    lora_layer = LoraLinear(
        adapter_name="default", 
        in_features=in_features,
        out_features=out_features,
        base_layer=qlinear,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        fan_in_fan_out=False,
        bias=bias,
        merge_weights=False
    )

    # 3. Initialize LoRA parameters
    torch.nn.init.kaiming_uniform_(lora_layer.lora_A["default"].weight, a=math.sqrt(5))
    torch.nn.init.zeros_(lora_layer.lora_B["default"].weight)

    return lora_layer


def prepare_reprofiling(model, rank):
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad = False

    for layer_idx, layer in enumerate(model.base_model.model.model.layers):
        for expert_idx in range(64):
            expert = layer.mlp.experts[expert_idx]
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    proj = getattr(expert, proj_name, None)
                    if not (hasattr(proj, "lora_A") and hasattr(proj, "lora_B")):
                        new_lora_layer = add_lora(proj, rank, 2*rank, 0.05)
                        setattr(expert, proj_name, new_lora_layer)
    
            for name, param in layer.mlp.experts[expert_idx].named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                 