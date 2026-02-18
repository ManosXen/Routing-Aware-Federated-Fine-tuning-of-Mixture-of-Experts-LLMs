from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import torch
from add_frozen_lora_adapter import add_or_update_lora_adapter
from update_lora_adapter import update_lora_adapter
from accumulate_converged_experts import accumulate_converged_experts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_trainable(model, trainable, rank, experts_per_layer=64, num_layers=16):
    if rank == 0:
        for layer_idx, layer in enumerate(model.base_model.layers):
            if layer_idx >= len(trainable):
                break
            for expert_idx in trainable[layer_idx]:
                for _, param in layer.mlp.experts[expert_idx].named_parameters():
                    param.requires_grad = True
    else:
        for layer_idx, layer in enumerate(model.base_model.model.model.layers):
            if layer_idx >= len(trainable):
                break
            for expert_idx in range(experts_per_layer):
                if expert_idx not in trainable[layer_idx]:
                    expert = layer.mlp.experts[expert_idx]
                    for proj_name in ("gate_proj", "up_proj", "down_proj"):
                        proj = getattr(expert, proj_name, None)
                        if proj is None:
                            print('Proj is none')
                            continue
                        base = proj.base_layer
                        # attach base in place of the wrapper
                        setattr(expert, proj_name, base)
                        frozen = all(not p.requires_grad for p in base.parameters())
                        if frozen==False:
                            print(f"Layer {layer_idx}, Expert {expert_idx}, {proj_name} frozen:", frozen)
                else:
                    for name, param in layer.mlp.experts[expert_idx].named_parameters():
                        #print(name)
                        if "lora" in name:
                            param.requires_grad = True

def build_model(model_name, trainable, qb, rank, load_prev_weights=True, client_lora_params=None, current_lora_weights=None, converged_experts=None, experts_per_layer=64, num_layers=16):
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

            fix_trainable(model, trainable, rank, experts_per_layer, num_layers)
            add_or_update_lora_adapter(model, current_lora_weights, client_lora_params, trainable, rank)
            #update_lora_adapter(model, converged_experts, current_lora_weights, client_lora_params, trainable, rank) -> weight merge, not good result
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer