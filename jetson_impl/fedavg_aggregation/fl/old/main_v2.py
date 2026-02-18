import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import argparse
import time
from activation_freeze import freeze_experts_activations
from gradient_freeze import freeze_experts_gradients
from esft import freeze_experts_esft
from train_v2 import train
from add_frozen_lora_adapter import add_or_update_lora_adapter
from prepare_reprofiling import prepare_reprofiling
import gc
import nvflare.client as flare

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
parser.add_argument("--dt_split", type=str, required=True,
                    help="Dataset")
parser.add_argument("--epochs", type=int, required=False, default=3,
                    help="Number of epochs")
parser.add_argument("--rank", type=int, required=False, default=0,
                    help="LoRA rank")
parser.add_argument("--batch_size", type=int, required=False, default=8,
                    help="Batch size")
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
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125").to('cuda')

if args.rank!=0:
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank*2,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")

if args.thr>args.fr:
    args.fr=args.thr  #for file name fix

model.current_round=0

lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"parameters before freeze: {lora_params}")

if model.current_round==0:
    if args.fc == "act_freeze":
        trainable, sorted_indx = freeze_experts_activations(model, args.fr, args.dt, args.rank, args.batch_size, 0)
    elif args.fc == "esft":
        trainable, sorted_indx = freeze_experts_esft(model, args.thr, args.dt, args.rank, args.batch_size, 0)
    else:
        trainable, sorted_indx = freeze_experts_gradients(model, args.fr, args.dt, args.rank, 0)
        for p in model.parameters():
            p.grad = None

    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters after freeze: {lora_params}")
#else:

    #received_params = fl_input.params
    #received_params = {k: v for k, v in received_params.items()}

    #lora_params=fl_input.lora_params

    #add_or_update_lora_adapter(model, received_params, lora_params, trainable)

    #del received_params, lora_params

    #Τώρα αλλάζω τους frozen experts σε κάθε γύρο. Αν θέλω να τους κρατήσω σταθερούς καθ' όλη την διάρκεια της εκπαίδευσης απλά το σβήνω
    if args.fc == "grad_freeze":

        prepare_reprofiling(model, args.rank)

        lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"parameters after_reprof: {lora_params}")

        trainable, sorted_indx = freeze_experts_gradients(model, args.fr, args.dt, fl_input.current_round)
        for p in model.parameters():
            p.grad = None


val_loss = train(model, args.dt, args.dt_split, model.current_round, args.fc, args.epochs, args.lr, args.tag, args.fr, args.fc, args.qb, args.rank, args.batch_size)

output_model = flare.FLModel(
    params={name: p.cpu() for name, p in model.named_parameters() if p.requires_grad},
    trainable=trainable,
    metrics={"val_loss": val_loss}
    #meta={"NUM_STEPS_CURRENT_ROUND": steps},
)
   
