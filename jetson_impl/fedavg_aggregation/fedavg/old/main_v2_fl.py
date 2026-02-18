import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from nvflare.app_common.abstract.fl_model import FLModel
import argparse
import time
from activation_freeze import freeze_experts_activations
from gradient_freeze import freeze_experts_gradients
from esft import freeze_experts_esft
from train_v2 import train
from add_frozen_lora_adapter import add_or_update_lora_adapter
from prepare_reprofiling import prepare_reprofiling
from python_filter import is_python_example
from prompt_formation import prompt_formation_and_tokenize_rosetta, prompt_formation_and_tokenize_meta_math, prompt_formation_and_tokenize_code
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
parser.add_argument("--start", type=int, required=False, 
                    help="Dataset start index")
parser.add_argument("--end", type=int, required=False, 
                    help="Dataset end index")
parser.add_argument("--epochs", type=int, required=False, default=3,
                    help="Number of epochs")
parser.add_argument("--rank", type=int, required=False, default=0,
                    help="LoRA rank")
parser.add_argument("--batch_size", type=int, required=False, default=8,
                    help="Batch size")
parser.add_argument("--range", type=int, required=False, default=-1,
                    help="Train dataset size. If not defined or -1 then, max possible")
parser.add_argument("--tag", type=str, default="", required=False,
                    help="Tag to differentiate result files of multiple training sessions with same hyperparams")
args = parser.parse_args()

if args.tag!="":
    args.tag="_"+args.tag

print("\n" + "=" * 60)
print("Parsed Training Configuration")
print("=" * 60)
print(f"  Freeze Config:")
print(f"    • Freeze criterion      : {args.fc}")
print(f"    • Freeze rate (fr)      : {args.fr}")
print(f"    • Threshold (thr)       : {args.thr}")
print()
print(f"  Quantization & Model:")
print(f"    • Quantization bits (qb): {args.qb}")
print(f"    • LoRA rank             : {args.rank}")
print()
print(f"  Training Params:")
print(f"    • Learning rate (lr)    : {args.lr}")
print(f"    • Epochs                : {args.epochs}")
print(f"    • Batch size            : {args.batch_size}")
print(f"    • Range                 : {args.range}")
print()
print(f"  Dataset Info:")
print(f"    • Dataset path          : {args.dt}")
print(f"    • Dataset split         : {args.dt_split}")
print(f"    • Start index           : {args.start}")
print(f"    • End index             : {args.end}")
print("=" * 60 + "\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

dataset=load_dataset(args.dt)
dataset = dataset["train"] if "train" in dataset else dataset

if args.dt_split=="none":
    if "codealpaca" in args.dt:
        dataset = dataset.filter(is_python_example, num_proc=4)
    dataset = dataset.select(range(args.start, args.end))

else:
    if args.dt=="databricks/databricks-dolly-15k":
        dataset = dataset.filter(lambda example: example["category"]==args.dt_split)
    if args.range == -1:
        total = len(dataset)
        train_size = int(0.9 * total)
        dataset = dataset.select(range(train_size))

dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

if "rosetta" in args.dt:
    prompt_formation_and_tokenize=prompt_formation_and_tokenize_rosetta
elif "meta-math" in args.dt:
    prompt_formation_and_tokenize=prompt_formation_and_tokenize_meta_math
elif "codealpaca" in args.dt:
    prompt_formation_and_tokenize=prompt_formation_and_tokenize_code
else:
    print("Error in prompt formation")
    exit

flare.init()
sys_info = flare.system_info()
client_name = sys_info["site_name"]
print(f"Client {client_name} initialized")

while flare.is_running():

    train_dataset.shuffle()
    val_dataset.shuffle()

    fl_input = flare.receive()

    current_round = fl_input.current_round

    if fl_input is None:
        continue
    print(f"current_round={current_round}")

    if current_round==0:
        if args.fc == "act_freeze":
            trainable, sorted_indx = freeze_experts_activations(model, args.fr, train_dataset, prompt_formation_and_tokenize, args.rank, args.batch_size, current_round)
        elif args.fc == "esft":
            trainable, sorted_indx = freeze_experts_esft(model, args.thr, train_dataset, prompt_formation_and_tokenize, args.rank, args.batch_size, current_round)
        else:
            trainable, sorted_indx = freeze_experts_gradients(model, args.fr, train_dataset, prompt_formation_and_tokenize, args.rank, current_round)
            for p in model.parameters():
                p.grad = None
        current_lora=trainable
    else:

        received_params = fl_input.params
        received_params = {k: v for k, v in received_params.items()}

        lora_params=fl_input.meta["lora_params"]

        add_or_update_lora_adapter(model, received_params, lora_params, current_lora, args.rank)

        for i in range(16):
            current_lora[i] = list(set(trainable[i]) | set(current_lora[i]))

        del received_params, lora_params

        #Τώρα αλλάζω τους frozen experts σε κάθε γύρο. Αν θέλω να τους κρατήσω σταθερούς καθ' όλη την διάρκεια της εκπαίδευσης απλά το σβήνω
        if args.fc == "grad_freeze":
            prepare_reprofiling(model, args.rank)
            trainable, sorted_indx = freeze_experts_gradients(model, args.fr, train_dataset, prompt_formation_and_tokenize, args.rank, current_round)
            for p in model.parameters():
                p.grad = None

    del fl_input
 
    val_loss = train(model, train_dataset, val_dataset, args.dt, prompt_formation_and_tokenize, current_round, args.fc, args.epochs, args.lr, args.tag, args.fr, args.fc, args.qb, args.rank, args.batch_size)
   
    output_model = flare.FLModel(
        params={name: p.cpu() for name, p in model.named_parameters() if p.requires_grad},
        current_round=current_round,
        metrics={"val_loss": val_loss},
        meta={"client_name" : client_name, "num_rows": len(train_dataset), "trainable": trainable, "rank": args.rank},
    )
    flare.send(output_model)
   
