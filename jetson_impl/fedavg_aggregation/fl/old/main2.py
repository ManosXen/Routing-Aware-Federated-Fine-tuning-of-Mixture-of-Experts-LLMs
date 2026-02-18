import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import argparse
import time
from activation_freeze import freeze_experts_activations
from gradient_freeze import freeze_experts_gradients
from esft import freeze_experts_esft
from train2 import train_and_memory_graph
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--fc", type=str, required=True, choices=["act_freeze", "grad_freeze", "esft", "none"],
                    help="Freeze choice: 'act_freeze', 'grad_freeze', or 'none'")
parser.add_argument("--fr", type=float, default=0, required=False,
                    help="freeze rate from 0 to 1")
parser.add_argument("--thr", type=float, default=0, required=False,
                    help="Threshold")
parser.add_argument("--qb", type=int, default=4, required=True, choices=[4, 8, 16],
                    help="Quantization bits")
parser.add_argument("--lr", type=float, default=1e-5, required=False,
                     help="Learning Rate")
parser.add_argument("--dt", type=str, required=True,
                    help="Dataset")
parser.add_argument("--epochs", type=int, required=False, default=3,
                    help="Number of epochs")
parser.add_argument("--rank", type=int, required=False, default=0,
                    help="LoRA rank")
parser.add_argument("--batch_size", type=int, required=False, default=8,
                    help="LoRA rank")
parser.add_argument("--tag", type=str, default="", required=False,
                    help="Tag to differentiate result files of multiple training sessions with same hyperparams")
args = parser.parse_args()

if args.tag!="":
    args.tag="_"+args.tag

print("="*20)
print(f"qb={args.qb}, fc={args.fc}, fr={args.fr}\ndt_path={args.dt}, lr={args.lr},rank={args.rank}\n{args.tag}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.qb==4:
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", quantization_config=qconfig).to('cuda')
    model = prepare_model_for_kbit_training(model).to(DEVICE)


elif args.qb==8:
    qconfig = BitsAndBytesConfig(
        load_in_8bit=True, 
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", quantization_config=qconfig)
    model = prepare_model_for_kbit_training(model)

else:
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", device_map="auto")
    print("Hello")
    print(model.hf_device_map)

if args.rank!=0:
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")

lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters before freeze: {lora_params}")

start_profiling=time.time()

if args.fc == "act_freeze":
    freeze_experts_activations(model, args.fr, args.dt, args.rank, args.batch_size)
elif args.fc == "grad_freeze":
    freeze_experts_gradients(model, args.fr, args.dt, args.rank)
    for p in model.parameters():
        if not p.requires_grad:
            p.grad = None
elif args.fc == "esft":
    freeze_experts_esft(model, args.thr, args.dt, args.rank, args.batch_size)

end_profiling=time.time()

gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()

print(f"parameters before freeze:{lora_params}")
if args.fc!='none':
    lora_params_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters after freeze:{lora_params_freeze}")
    print(f"parameters saved:{(lora_params-lora_params_freeze)/lora_params*100:.2f}")

start_training=time.time()

if args.thr>args.fr:
    args.fr=args.thr  #for file name fix

train_and_memory_graph(model, args.dt, args.fc, args.epochs, args.lr, args.tag, args.fr, args.fc, args.qb, args.rank, args.batch_size)

end_training = time.time()

profiling_time=end_profiling-start_profiling
training_time=end_training-start_training
total_time=profiling_time+training_time

print(f"Elapsed time for profiling: {profiling_time:.2f}\n")
print(f"Elapsed time for training: {training_time:.2f}\n")
print(f"Elapsed time overall: {total_time:.2f}\n")

with open(f"elapsed_time_{args.qb}_{args.fc}_{args.fr}_{args.rank}_{args.lr}_{args.wd}{args.tag}.txt", "w") as f:
    f.write(f"Elapsed time for profiling: {profiling_time:.2f}\n")
    f.write(f"Elapsed time for training: {training_time:.2f}\n")
    f.write(f"Elapsed time overall: {total_time:.2f}\n")
    f.write(f"LoRA parameters before freeze:{lora_params}\n")
    if args.fc=='act_freeze' or args.fc=='grad_freeze':
        f.write(f"LoRA parameters after freeze:{lora_params_freeze}\n")
        f.write(f"LoRA parameters saved:{(lora_params-lora_params_freeze)/lora_params*100:.2f}\n")

