import torch
from peft.tuners.lora import Linear as LoraLinear
import bitsandbytes as bnb
import bitsandbytes.functional as bnb_F

device='cuda'

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

def accumulate_lora_into_base(expert):
    try:
        base_layer = expert.base_layer
        dtype = expert.lora_A['default'].weight.dtype
        if hasattr(expert, 'scaling'):
            scaling = expert.scaling["default"]
        else:
            scaling = expert.lora_alpha / expert.r

        #print(f"Scaling: {scaling}")
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        identity = torch.eye(in_features, device=device, dtype=dtype)

        #print(f"Quantized: {base_layer.weight}")
        with torch.no_grad():
            out = base_layer(identity)        # shape [in, out]
            if base_layer.bias is not None:
                out -= base_layer.bias        # shape [out]
            base_weight_dequantized = out.T 
        
        #print(f"Dequantized: {base_weight_dequantized}")
        lora_A = expert.lora_A['default'].weight
        lora_B = expert.lora_B['default'].weight

        delta_w = (lora_B @ lora_A) * scaling
        #print(f"ΔW {delta_w}")
        merged_weight = base_weight_dequantized + delta_w
        #print(f"Merged Weights: {merged_weight}")

        base_dtype=base_layer.weight.dtype
        #print(f"Base layer dtype: {base_dtype}")

        mw_fl16 = merged_weight.to(torch.float16)
        del merged_weight
        
        if base_dtype==torch.int8:
            CA, CS, _ = bnb_F.int8_vectorwise_quant(A=mw_fl16)
            #print(f"CA, SCA : {CA}, {CS}")
            new_layer = bnb.nn.Linear8bitLt(
                in_features,
                out_features,
                threshold=0.0,
                has_fp16_weights=expert.base_layer.weight.has_fp16_weights,
                bias=base_layer.bias is not None,
            ).to('cuda')

            new_layer.weight = bnb.nn.Int8Params(
                data=CA,
                SCB=CS, 
                requires_grad=False, 
                has_fp16_weights=expert.base_layer.weight.has_fp16_weights
            ).to('cuda')

            new_layer.state.SCB = CS
        
        elif base_dtype==torch.nf4:
            CA, SCA = bnb_F.quantize_nf4(A=merged_weight, compress_statistics=True)

            new_layer = bnb.nn.LinearNF4(
                in_features,
                out_features,
                threshold=0.0,
                has_fp16_weights=expert.base_layer.weight.has_fp16_weights,
                compute_dtype=torch.bfloat16,
                bias=base_layer.bias is not None,
            ).to('cuda')

            new_layer.weight = bnb.nn.Params4bit(
                data=CA,
                quant_state=SCA,
                quant_type="nf4",
                requires_grad=False, 
                has_fp16_weights=expert.base_layer.weight.has_fp16_weights
            ).to('cuda')

        return new_layer
        
    except Exception as e:
        print(f"!!! ERROR during merge: {e}")
        raise e

def accumulate_expert_lora(expert):
    """
Signature simplified: The LoraLinear layer already has its weights.
    """
    #print(f'Accumulating (merging) LoRA for expert...')
    expert.gate_proj = accumulate_lora_into_base(expert.gate_proj)
    expert.up_proj = accumulate_lora_into_base(expert.up_proj)
    expert.down_proj = accumulate_lora_into_base(expert.down_proj)

    # for proj_name in ("gate_proj", "up_proj", "down_proj"):
    #     proj = getattr(expert, proj_name, None)
    #     if proj is None:
    #         print('Proj is none')
    #         continue
    #     base = proj.base_layer
    #     # attach base in place of the wrapper
    #     setattr(expert, proj_name, base)

    #print(f"New Expert: {expert}")

    frozen = all(not p.requires_grad for p in expert.parameters())
    if frozen==False:
        print(f"Frozen:", frozen)

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



def update_lora_adapter(model, converged_experts, received_params, lora_params, trainable, rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(model)

    for layer_idx, layer in enumerate(model.base_model.model.model.layers):
        if layer_idx >= len(lora_params):
            break

        for expert_idx in lora_params[layer_idx]:
            gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B = [
                p.to(device) for p in find_lora_weights_received_params(received_params, layer_idx, expert_idx)
            ]
            #lora_params_before = len([name for name, p in model.named_parameters() if "lora" in name])
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

            #lora_params_total = len([name for name, p in model.named_parameters() if "lora" in name])
            
            if (expert_idx+layer_idx*64) in converged_experts: 
                accumulate_expert_lora(layer.mlp.experts[expert_idx])
                #lora_params_conv = len([name for name, p in model.named_parameters() if "lora" in name])
                #print(f"LoRA params before: {lora_params_before}\nLoRA params total: {lora_params_total}\nLoRA params after conv: {lora_params_conv}")
    torch.cuda.empty_cache()