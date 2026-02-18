import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from prompt_formation import (
    prompt_formation_and_tokenize_rosetta,
    prompt_formation_and_tokenize_meta_math
)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
effective_batch_size = 32


def evaluate(model, val_loader, batch_size):
    model.eval()
    val_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            # Send batch to the same device as model input embeddings
            first_device = next(model.parameters()).device
            batch = {k: v.to(first_device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            del outputs, batch
            count += 1
    model.train()
    eff_batches = count/(effective_batch_size / batch_size)
    return val_loss / eff_batches


def train(model, train_dataset, val_dataset, dataset_path, prompt_formation_and_tokenize, round, train_name, epochs, lr, fr, fc, qb, rank, batch_size):

    train_dataset = train_dataset.map(prompt_formation_and_tokenize, batched=False)
    val_dataset = val_dataset.map(prompt_formation_and_tokenize, batched=False)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")
    loader_train = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True, pin_memory=True, num_workers=4)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    num_batches = len(loader_train)

    losses=list()

    mem_stats = []

    accumulation_steps=effective_batch_size // batch_size
    num_gpus = torch.cuda.device_count()

    print(f"Training using {num_gpus} GPU(s).")

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(loader_train):
            first_device = next(model.parameters()).device
            batch = {k: v.to(first_device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elif (batch_idx + 1) == len(loader_train):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f"Loss: {loss.item():.4f} | Batch {batch_idx}/{num_batches} | Epoch {epoch+1}")

            del batch, loss, outputs

        torch.cuda.empty_cache()
        val_loss = evaluate(model, loader_val, batch_size)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
    
    return val_loss