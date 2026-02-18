import json
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
import sys

dataset = sys.argv[1]
num_clients = int(sys.argv[2])
alpha = float(sys.argv[3])
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

def get_dirichlet_indices(global_indices, labels, num_clients, alpha, seed):
    """Generates non-IID indices using Dirichlet distribution while preserving global pointers."""
    np.random.seed(seed)

    label_to_global_idx = defaultdict(list)
    for g_idx, label in zip(global_indices, labels):
        label_to_global_idx[label].append(g_idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for label in label_to_global_idx.keys():
        indices = label_to_global_idx[label]
        np.random.shuffle(indices)
        
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(indices)).astype(int)
        
        diff = len(indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        print(f"Label {label} distribution: {counts}")

        current = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(indices[current : current + count])
            current += count

    return client_indices

dt = load_dataset(dataset, split="train")    

dataset_label_map = {
    "nyu-mll/multi_nli": "label",
    "jet-ai/social_i_qa": "label",
    "allenai/winogrande": "answer",
    "google/boolq": "answer",
    "tau/commonsense_qa": "answerKey",
    "baber/piqa": "label"
}

all_global_indices = list(range(len(dt)))
if len(all_global_indices) > 120000:
    print(f"Dataset too large ({len(all_global_indices)}). Sampling 120k indices...")
    sampled_indices = random.sample(all_global_indices, 120000)
else:
    sampled_indices = all_global_indices

sampled_labels = [dt[i][dataset_label_map[dataset]] for i in sampled_indices]

exp2_indices = get_dirichlet_indices(sampled_indices, sampled_labels, num_clients, alpha, SEED)

class_counts_per_client = defaultdict(lambda: np.zeros(num_clients))


global_label_lookup = dict(zip(sampled_indices, sampled_labels))

for client_id, indices in enumerate(exp2_indices):
    counts = Counter([global_label_lookup[idx] for idx in indices])
    for label, count in counts.items():
        class_counts_per_client[label][client_id] = count

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
clients = [f"Client {i}" for i in range(num_clients)]
bottom = np.zeros(num_clients)
sorted_classes = sorted(class_counts_per_client.keys())

for label in sorted_classes:
    counts = class_counts_per_client[label]
    ax.bar(clients, counts, bottom=bottom, label=f"Label {label}")
    bottom += counts

safe_dataset_path = dataset.replace("/", "_")
ax.set_ylabel('Number of Samples')
ax.set_title(f'Label Distribution across Clients (alpha={alpha})')
ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'/files/label_distribution_{safe_dataset_path}_{alpha}.png')

# 5. Save JSON
data = {"clients": exp2_indices}
with open(f"/files/{safe_dataset_path}_{alpha}_data_split.json", "w") as f:
    json.dump(data, f, indent=4) 