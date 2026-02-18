import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from prompt_formation import prompt_formation_and_tokenize_rosetta, prompt_formation_and_tokenize_meta_math, prompt_formation_and_tokenize_dolly15k


tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
effective_batch_size=32

def evaluate(model, val_loader):
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
    return val_loss / count

#train_name is used for naming the memory graph and model weights appropriately
def train(model, train_dataset, val_dataset, dataset_path, prompt_formation_and_tokenize, round, train_name, epochs, lr, tag, fr, fc, qb, rank, batch_size):

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

    for epoch in range(epochs):  # epochs
        for batch_idx, batch in enumerate(loader_train):

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            if (batch_idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elif (batch_idx + 1) == len(loader_train):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            mem_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2    # MB
            mem_stats.append((epoch, batch_idx, mem_allocated, mem_reserved))

            losses.append(outputs.loss.item())

            if batch_idx % 100 == 0:
                print(f"Loss:{outputs.loss} Batch {batch_idx}/{num_batches} (epoch {epoch+1})")

            del batch, loss, outputs

        torch.cuda.empty_cache()

        val_loss = evaluate(model, loader_val)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        trainable_weights = {name: param for name, param in model.named_parameters() if param.requires_grad}
        safe_dataset_path = dataset_path.replace("/", "_")
        base_filename = f"model_{qb}_{fc}_{fr}_{round}_{rank}_{safe_dataset_path}_{lr}_{tag}"
        torch.save(trainable_weights, f"{base_filename}.pt")

        print(f"Saved {len(trainable_weights)} trainable parameter tensors.")
        
        torch.cuda.empty_cache()
    
    steps = [f"{e}-{b}" for e, b, _, _ in mem_stats]
    allocated = [a for _, _, a, _ in mem_stats]
    reserved = [r for _, _, _, r in mem_stats]

    # Compute statistics
    avg_allocated = np.mean(allocated)
    max_allocated = np.max(allocated)
    avg_reserved = np.mean(reserved)
    max_reserved = np.max(reserved)

    # # --- Validation loss per epoch ---
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(1, len(losses)+1), losses, marker="o", color="red")
    # plt.title(f"Validation Loss per Epoch {train_name}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Validation Loss")
    # plt.grid(True)
    # name=f"val_loss_{qb}_{fc}_{fr}_{round}_{rank}_{lr}_{tag}.png"
    # plt.savefig(name, bbox_inches='tight')

    # with open(f"gpu_memory_{qb}_{fc}_{fr}_{round}_{rank}_{lr}_{tag}.txt", "w") as f:
    #     f.write(f"Avg Allocated Memory: {avg_allocated:.2f}\n")
    #     f.write(f"Max Allocated Memory: {max_allocated:.2f}\n\n")
    #     f.write(f"Avg Reserved Memory: {avg_reserved:.2f}\n")
    #     f.write(f"Max Reserved Memory: {max_reserved:.2f}\n")

    return val_loss



