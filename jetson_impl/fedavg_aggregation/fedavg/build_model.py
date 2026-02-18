from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_lora_weights_received_params(received_params, layer, expert):
    base = f"base_model.model.model.layers.{layer}.mlp.experts.{expert}"

    gate_proj_A = received_params[f"{base}.gate_proj.lora_A.default.weight"]
    gate_proj_B = received_params[f"{base}.gate_proj.lora_B.default.weight"]

    up_proj_A = received_params[f"{base}.up_proj.lora_A.default.weight"]
    up_proj_B = received_params[f"{base}.up_proj.lora_B.default.weight"]

    down_proj_A = received_params[f"{base}.down_proj.lora_A.default.weight"]
    down_proj_B = received_params[f"{base}.down_proj.lora_B.default.weight"]

    return gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B


def build_model(model_name, qb, rank, load_prev_weights=True, client_lora_params=None, current_lora_weights=None, converged_experts=None, experts_per_layer=64, num_layers=16):
    if qb==4:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=qconfig)
        model = prepare_model_for_kbit_training(model)


    elif qb==8:
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True, 
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=qconfig)
        model = prepare_model_for_kbit_training(model)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if rank!=0:

        config = LoraConfig(
            r=rank,
            lora_alpha=rank*2,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)

        if load_prev_weights:
            for layer_idx, layer in enumerate(model.base_model.model.model.layers):
                for expert_idx in range(experts_per_layer):
                    gate_proj_A, gate_proj_B, up_proj_A, up_proj_B, down_proj_A, down_proj_B = [
                        p.to(DEVICE) for p in find_lora_weights_received_params(current_lora_weights, layer_idx, expert_idx)
                    ]

                    expert = layer.mlp.experts[expert_idx]
                    with torch.no_grad():
                        expert.gate_proj.lora_A["default"].weight.copy_(gate_proj_A)
                        expert.gate_proj.lora_B["default"].weight.copy_(gate_proj_B)
                        expert.up_proj.lora_A["default"].weight.copy_(up_proj_A)
                        expert.up_proj.lora_B["default"].weight.copy_(up_proj_B)
                        expert.down_proj.lora_A["default"].weight.copy_(down_proj_A)
                        expert.down_proj.lora_B["default"].weight.copy_(down_proj_B)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer