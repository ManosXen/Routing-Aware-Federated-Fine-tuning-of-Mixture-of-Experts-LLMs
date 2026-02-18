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


def evaluate(model, val_loader):
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
    return val_loss / max(count, 1)


def train_and_memory_graph(model, dataset_path, train_name, epochs, lr, tag, fr, fc, qb, rank, batch_size):

    subset_size = 65000
    dataset = load_dataset(dataset_path)["train"].shuffle(seed=42).select(range(subset_size))
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset, val_dataset = dataset["train"], dataset["test"]

    # Choose tokenizer function
    if "rosetta" in dataset_path:
        prompt_formation_and_tokenize = prompt_formation_and_tokenize_rosetta
    elif "meta-math" in dataset_path:
        prompt_formation_and_tokenize = prompt_formation_and_tokenize_meta_math
    else:
        raise ValueError("Unknown dataset for prompt formation")

    train_dataset = train_dataset.map(prompt_formation_and_tokenize, batched=False)
    val_dataset = val_dataset.map(prompt_formation_and_tokenize, batched=False)
    max_len = max(len(sequence) for sequence in train_dataset["input_ids"])
    print(f"Max input length: {max_len}")

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest")
    loader_train = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator,
                              shuffle=True, pin_memory=True, num_workers=4)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator,
                            shuffle=True, pin_memory=True, num_workers=4)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    num_batches = len(loader_train)
    losses, val_losses = [], []
    best_loss = float("inf")

    mem_stats = []  # Will hold (epoch, batch_idx, gpu_id, allocated, reserved)
    accumulation_steps = effective_batch_size / batch_size
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

            # --- Record memory stats for all GPUs ---
            torch.cuda.synchronize()
            for gpu_id in range(num_gpus):
                mem_alloc = torch.cuda.memory_allocated(gpu_id) / 1024 ** 2  # MB
                mem_resv = torch.cuda.memory_reserved(gpu_id) / 1024 ** 2
                mem_stats.append((epoch, batch_idx, gpu_id, mem_alloc, mem_resv))

            losses.append(loss.item())

            if batch_idx % 100 == 0:
                print(f"Loss: {loss.item():.4f} | Batch {batch_idx}/{num_batches} | Epoch {epoch+1}")

            del batch, loss, outputs

        torch.cuda.empty_cache()

        # Evaluate validation loss
        val_loss = evaluate(model, loader_val)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            trainable_weights = {n: p for n, p in model.named_parameters() if p.requires_grad}
            safe_dataset_path = dataset_path.replace("/", "_")
            base_filename = f"model_{qb}_{fc}_{fr}_{rank}_{safe_dataset_path}_{lr}_{tag}"
            torch.save(trainable_weights, f"{base_filename}.pt")
            print(f"Saved {len(trainable_weights)} trainable parameter tensors.")
            best_loss = val_loss

        torch.cuda.empty_cache()

    # --- Aggregate and write stats per GPU ---
    mem_stats = np.array(mem_stats, dtype=object)
    steps = [f"{e}-{b}" for e, b, _, _, _ in mem_stats]
    gpu_ids = np.unique(mem_stats[:, 2])

    with open(f"gpu_memory_{qb}_{fc}_{fr}_{rank}_{lr}_{tag}.txt", "w") as f:
        for gpu_id in gpu_ids:
            gpu_rows = mem_stats[mem_stats[:, 2] == gpu_id]
            allocated = gpu_rows[:, 3].astype(float)
            reserved = gpu_rows[:, 4].astype(float)
            avg_alloc, max_alloc = np.mean(allocated), np.max(allocated)
            avg_resv, max_resv = np.mean(reserved), np.max(reserved)
            f.write(f"=== GPU {gpu_id} ===\n")
            f.write(f"Avg Allocated Memory: {avg_alloc:.2f} MB\n")
            f.write(f"Max Allocated Memory: {max_alloc:.2f} MB\n")
            f.write(f"Avg Reserved Memory: {avg_resv:.2f} MB\n")
            f.write(f"Max Reserved Memory: {max_resv:.2f} MB\n\n")