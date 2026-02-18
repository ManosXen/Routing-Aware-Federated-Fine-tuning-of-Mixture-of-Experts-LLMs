import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from prompt_formation import prompt_formation_and_tokenize_rosetta, prompt_formation_and_tokenize_meta_math


tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def train_and_memory_graph(model, dataset_path, train_name, epochs, lr, tag, fr, fc, qb, rank):

    dataset=load_dataset(dataset_path)

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)  # 90% train, 10% val
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    if "rosetta" in dataset_path:
        prompt_formation_and_tokenize=prompt_formation_and_tokenize_rosetta
    elif "meta-math" in dataset_path:
        prompt_formation_and_tokenize=prompt_formation_and_tokenize_meta_math
    else:
        print("Error in prompt formation")
        return

    train_dataset = train_dataset.map(prompt_formation_and_tokenize, batched=False)
    val_dataset = val_dataset.map(prompt_formation_and_tokenize, batched=False)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(tokenizer)
    loader_train = DataLoader(train_dataset, batch_size=8, collate_fn=collator, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=8, collate_fn=collator, shuffle=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    num_batches = len(loader_train)

    losses=list()
    val_losses=list()
    best_loss=100

    mem_stats = []
    

    for epoch in range(epochs):  # epochs
        for batch_idx, batch in enumerate(loader_train):

            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mem_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2  # MB
            mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**2    # MB
            mem_stats.append((epoch, batch_idx, mem_allocated, mem_reserved))

            losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"Loss:{loss} Batch {batch_idx}/{num_batches} (epoch {epoch+1})")

            del batch, loss, outputs

        torch.cuda.empty_cache()

        val_loss = evaluate(model, loader_val)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
        if(val_loss<best_loss):
            model.save_pretrained(f"model_{qb}_{fc}_{fr}_{rank}_{dataset_path}_{lr}_{tag}")
            best_loss=val_loss
        
        torch.cuda.empty_cache()
    
    steps = [f"{e}-{b}" for e, b, _, _ in mem_stats]
    allocated = [a for _, _, a, _ in mem_stats]
    reserved = [r for _, _, _, r in mem_stats]

    # Compute statistics
    avg_allocated = np.mean(allocated)
    max_allocated = np.max(allocated)
    avg_reserved = np.mean(reserved)
    max_reserved = np.max(reserved)

    # plt.figure(figsize=(12,6))
    # plt.plot(steps, allocated, label="Allocated (MB)")
    # plt.plot(steps, reserved, label="Reserved (MB)")

    # # Add average and max lines
    # plt.axhline(avg_allocated, color='blue', linestyle='--', linewidth=1, label=f"Avg Allocated ({avg_allocated:.1f} MB)")
    # plt.axhline(max_allocated, color='blue', linestyle=':', linewidth=1, label=f"Max Allocated ({max_allocated:.1f} MB)")
    # plt.axhline(avg_reserved, color='orange', linestyle='--', linewidth=1, label=f"Avg Reserved ({avg_reserved:.1f} MB)")
    # plt.axhline(max_reserved, color='orange', linestyle=':', linewidth=1, label=f"Max Reserved ({max_reserved:.1f} MB)")

    # plt.xticks(rotation=90, fontsize=6)
    # plt.legend(fontsize=8)
    # plt.title(f"GPU Memory Usage During Training {train_name}")
    # plt.ylabel("MB")
    # plt.xlabel("Step")
    # plt.tight_layout()

    # name = f"memory_usage_{train_name}.png"
    # plt.savefig(name, bbox_inches='tight')

    # window = 50
    # smoothed_losses = np.convolve(losses, np.ones(window)/window, mode='valid')

    # plt.figure(figsize=(10, 5))
    # plt.plot(losses, color="lightblue", alpha=0.5, label="Raw batch loss")
    # plt.plot(range(window-1, len(losses)), smoothed_losses, color="blue", label=f"Smoothed (window={window})")
    # plt.title(f"Training Loss per Batch {train_name}")
    # plt.xlabel("Batch step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)
    # name=f"batch_loss_smouthed_{qb}_{fc}_{fr}_{lr}_{wd}{tag}.png"
    # plt.savefig(name, bbox_inches='tight')


    # --- Validation loss per epoch ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(val_losses)+1), val_losses, marker="o", color="red")
    plt.title(f"Validation Loss per Epoch {train_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    name=f"val_loss_{qb}_{fc}_{fr}_{rank}_{lr}_{tag}.png"
    plt.savefig(name, bbox_inches='tight')

    with open(f"gpu_memory_{qb}_{fc}_{fr}_{rank}_{lr}_{tag}.txt", "w") as f:
        f.write(f"Avg Allocated Memory: {avg_allocated:.2f}\n")
        f.write(f"Max Allocated Memory: {max_allocated:.2f}\n\n")
        f.write(f"Avg Reserved Memory: {avg_reserved:.2f}\n")
        f.write(f"Max Reserved Memory: {max_reserved:.2f}\n")



