import torch
from peft.tuners.lora import Linear as LoraLinear

def accumulate_lora_into_base(base_linear, A, B, r, alpha):
    W = base_linear.weight
    print(W)
    scale = alpha / r
    delta = (B @ A) * scale
    W += delta
    

def accumulate_expert_lora(expert, gate_A, gate_B, up_A, up_B, down_A, down_B, r, alpha):
    print('Here')
    accumulate_lora_into_base(expert.gate_proj,  gate_A,  gate_B,  r, alpha)
    accumulate_lora_into_base(expert.up_proj,    up_A,    up_B,    r, alpha)
    accumulate_lora_into_base(expert.down_proj,  down_A,  down_B,  r, alpha)

def find_lora_weights_received_params(received_params, layer, expert):
    base = f"base_model.model.model.layers.{layer}.mlp.experts.{expert}"

    gate_proj_A = received_params[f"{base}.gate_proj.lora_A.default.weight"]
    gate_proj_B = received_params[f"{base}.gate_proj.lora_B.default.weight"]

    up_proj_A = received_params[f"{base}.up_proj.lora_A.default.weight"]
    up_proj_B = received_params[f"{base}.up_proj.lora_B.default.weight"]

    down_proj_A = received_params[f"{base}.down_proj.lora_A.default.weight"]
    down_proj_B = received_params[f"{base}.down_proj.lora_B.default.weight"]

    return gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B



def accumulate_converged_experts(model, converged_experts, received_params, lora_params, trainable, rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(model)
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx >= len(lora_params):
            break

        for expert_idx in lora_params[layer_idx]:
            if (expert_idx+layer_idx*64) in converged_experts:
                print(f'Converged {expert_idx+layer_idx*64}')
                gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B = [
                    p.to(device) for p in find_lora_weights_received_params(received_params, layer_idx, expert_idx)
                ]

                if expert_idx not in trainable[layer_idx]:
                    accumulate_expert_lora(layer.mlp.experts[expert_idx],
                                        gate_proj_A, gate_proj_B,
                                        up_proj_A, up_proj_B,
                                        down_proj_A, down_proj_B,
                                        rank, 2*rank)