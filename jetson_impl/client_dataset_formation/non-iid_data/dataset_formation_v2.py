import json
import numpy as np
from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
import sys
import os

# --- Configuration & Titles ---
DATASET_TITLES = {
    "piqa": [
        "Cooking & Meal Preparation", "Kitchen Supplies & Storage", "Surface Finishing & Painting",
        "Construction & Crafting materials", "Workspace Organization & DIY", "Baking & Dough Chemistry",
        "Life Hacks & Preservation", "Tools & Mechanical Manipulation", "Electronics & Digital Media",
        "Confectionery & Sweets", "Gardening & Environment Control", "Cleaning & Stain Removal"
    ],
    "commonsense_qa": [
        "Socializing & Recreation", "Physical Activity & Body States", "Academic & Social Concepts",
        "Domestic Environments", "Public Infrastructure & Places", "Subjective Narratives & Feelings",
        "Family Life & Parenting", "Crime, Law & Conflict", "Nature & Physical Elements",
        "Food & Dining", "Animals & Biology", "Commerce & Household Objects"
    ],
    "boolq": [
        "Gun Control & Firearms Law", "Zoology & Species Classification", "TV Series Production & Air Dates",
        "Immigration & Citizenship", "Video Games & Consoles", "US Professional Sports (NBA/NFL)",
        "Biology & Physiology", "Film & Literature Adaptations", "TV Plotlines & Character Fates",
        "International Soccer (FIFA/UEFA)", "US Politics & Constitution", "Geography & Infrastructure",
        "Business & Corporate Brands"
    ],
    "winogrande": [
        "Social Scenarios (Female A)", "Everyday Objects & Physics", "Social Scenarios (Male A)",
        "Clothing & Appearance", "Buildings & Locations", "Spatial Dimensions & Fit",
        "Social Scenarios (Male B)", "Liquids & Volume", "Food, Dining & Taste",
        "Social Scenarios (Female B)"
    ],
    "social_i_qa": [
        "Pets & Playtime", "Business & Appliances", "Lost Belongings", "Celebrations & Birthdays",
        "School & Emotions", "Family Relations", "Pet Care & Feeding", "Extended Family",
        "Domestic Rules", "Daily Routines", "Career & Applications", "General Pet Interactions",
        "Mother-Daughter Dynamics", "Gifts & Hosting", "Competition & Conflict", "Professional Actions"
    ],
    "race": [
        "Family & Relationships", "School & Classroom", "Culture & Language", "Fiction & Dialogue",
        "Biographies & History", "Personal Growth", "Tech & Digital Media", "Chinese Culture",
        "Education & Parenting", "Environment & Ecology", "Zoology & Animals", "Travel & Tourism",
        "Health & Biology"
    ]
}

try:
    dataset_name = sys.argv[1]
    dataset_file = sys.argv[2]
    num_clients = int(sys.argv[3])
    alpha = float(sys.argv[4])
except IndexError:
    print("Usage: python script.py <dataset_name> <cluster_json> <num_clients> <alpha>")
    sys.exit(1)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def dirichlet_distribution(clusters, num_clients, alpha, seed):
    np.random.seed(seed)
    client_indices = [[] for _ in range(num_clients)]
    for cluster_id, cluster in enumerate(clusters):
        indices = np.array(cluster, dtype='int')
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(indices)).astype(int)
        diff = len(indices) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1
        current = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(indices[current : current + count])
            current += count        
    return client_indices

# Load Data
with open(dataset_file, 'r') as f:
    clusters = json.load(f)['cluster_idx']

global_cluster_lookup = {idx: c_id for c_id, ids in enumerate(clusters) for idx in ids}
exp2_indices = dirichlet_distribution(clusters, num_clients, alpha, SEED)

cluster_counts_per_client = defaultdict(lambda: np.zeros(num_clients))
for client_id, indices in enumerate(exp2_indices):
    client_cluster_ids = [global_cluster_lookup[idx] for idx in indices if idx in global_cluster_lookup]
    for cluster_id, count in Counter(client_cluster_ids).items():
        cluster_counts_per_client[cluster_id][client_id] = count

#Plotting
fig, ax = plt.subplots(figsize=(12, 7))
clients_labels = [f"Client {i}" for i in range(num_clients)]
bottom = np.zeros(num_clients)
sorted_cluster_ids = sorted(cluster_counts_per_client.keys())

cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, len(sorted_cluster_ids))]

dataset_specific_titles = DATASET_TITLES.get(dataset_name, [])

for cluster_id in sorted_cluster_ids:
    counts = cluster_counts_per_client[cluster_id]
    if cluster_id < len(dataset_specific_titles):
        label_text = dataset_specific_titles[cluster_id]
    else:
        label_text = f"Cluster {cluster_id}"   
    ax.bar(clients_labels, counts, bottom=bottom, label=label_text, color=colors[cluster_id])
    bottom += counts

safe_name = dataset_name.replace("/", "_")
ax.set_ylabel('Number of Samples')
ax.set_title(f'Cluster Distribution: {dataset_name} (alpha={alpha})')
ax.legend(title="Cluster Topics", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

os.makedirs("files", exist_ok=True)
plt.tight_layout()
plt.savefig(f'files/client_distr_{safe_name}_{alpha}.png')

# 4. Save JSON (Fixed with int conversion for NumPy types)
serializable = [[int(idx) for idx in c_list] for c_list in exp2_indices]
with open(f"files/{safe_name}_{alpha}_split.json", "w") as f:
    json.dump({"clients": serializable}, f, indent=4)