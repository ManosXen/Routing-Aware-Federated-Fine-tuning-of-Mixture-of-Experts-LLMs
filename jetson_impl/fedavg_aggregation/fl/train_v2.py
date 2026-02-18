import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from prompt_formation import *
from no_trainable_params_exception import NoTrainableParams

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
effective_batch_size=32

def evaluate(model, val_loader, batch_size):
    model.eval()
    val_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            del outputs, batch
            count += 1
    model.train()
    eff_batches = count/(effective_batch_size / batch_size)
    return val_loss / eff_batches

#train_name is used for naming the memory graph and model weights appropriately
def train(model, train_dataset, dataset_path, prompt_formation_and_tokenize, round, train_name, epochs, lr, fr, fc, qb, rank, batch_size):

    #print("Inside Training", flush=True)
    train_dataset = train_dataset.map(prompt_formation_and_tokenize, batched=False)
    #val_dataset = val_dataset.map(prompt_formation_and_tokenize, batched=False)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    #val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")
    loader_train = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4, persistent_workers=True)
    #loader_val = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4)

    #print("Fixed dataset loaders", flush=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("No trainable parameters found. NoTrainableParams exception.", flush=True)
        raise NoTrainableParams()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    #print("Optimizer OK", flush=True)
    model.train()
    #print("Model train OK", flush=True)
    num_batches = len(loader_train)

    losses=list()

    mem_stats = []

    accumulation_steps=effective_batch_size // batch_size
    #print("Outside Training", flush=True)
    for epoch in range(epochs):  # epochs
        for batch_idx, batch in enumerate(loader_train):

            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            #print("batch to device", flush=True)
            outputs = model(**batch)
            #print("calculated output", flush=True)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            #print("backward", flush=True)
            if (batch_idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elif (batch_idx + 1) == len(loader_train):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            #print("Optimizer Step", flush=True)
            mem_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2    # MB
            mem_stats.append((epoch, batch_idx, mem_allocated, mem_reserved))
            #print("Memory", flush=True)
            losses.append(outputs.loss.item())

            if batch_idx % 100 == 0:
                print(f"Loss:{outputs.loss} Batch {batch_idx}/{num_batches} (epoch {epoch+1})", flush=True)

            del batch, loss, outputs

        torch.cuda.empty_cache()

        # val_loss = evaluate(model, loader_val, batch_size)
        # print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}", flush=True)

        trainable_weights = {name: param for name, param in model.named_parameters() if param.requires_grad}

    del optimizer
    val_loss=0.3 #placeholder
    return val_loss, mem_stats



