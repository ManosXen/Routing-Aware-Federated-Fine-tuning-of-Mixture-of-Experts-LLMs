from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
import argparse
import torch
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_file", type=str, required=True,
                    help="Path to LoRA Adapter")
parser.add_argument("--save_path", type=str, required=True,
                    help="Path to save merged model")
parser.add_argument("--qb", type=int, default=4, choices=[4, 8, 16])
parser.add_argument("--rank", type=int, default=8)
args = parser.parse_args()

base_model_name = "allenai/OLMoE-1B-7B-0125"

if args.qb==4:
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", quantization_config=qconfig).to('cuda')

elif args.qb==8:
    qconfig = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", quantization_config=qconfig)
else:
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125").to('cuda')

if args.rank!=0:
    config = LoraConfig(
        r=args.rank,
        lora_alpha=2*args.rank,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, config)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

saved_trainable_weights = torch.load(args.adapter_file, map_location="cuda")
model.load_state_dict(saved_trainable_weights, strict=False)

os.makedirs(args.save_path, exist_ok=True)

model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)

base_tokenizer_dir = os.path.join(
    os.environ.get("TRANSFORMERS_CACHE", "~/.cache/huggingface/transformers"),
    f"models--{base_model_name.replace('/', '--')}"
)
for root, _, files in os.walk(base_tokenizer_dir):
    if "tokenizer.model" in files:
        src = os.path.join(root, "tokenizer.model")
        dst = os.path.join(args.save_path, "tokenizer.model")
        shutil.copyfile(src, dst)
        print(f"Copied tokenizer.model from {src} to {dst}")
        break