import matplotlib.pyplot as plt
import sys
import json
import os
import re
import numpy as np

def get_round_from_filename(filename):
    """
    Extracts the round number from filenames like:
    piqa_act_thr_params_13_2026-01-23T02-09-46.125984.json
    Target pattern: params_{ROUND}_2026
    """
    match = re.search(r'params_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def get_accuracy_from_json(filepath):
    """
    Reads the JSON and attempts to find an accuracy metric.
    Handles nested structures like: results -> piqa -> acc,none
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 1. Check for specific nested structure from your snippet
        if "results" in data:
            results = data["results"]
            # Iterate over tasks (e.g., 'piqa')
            for task_name, metrics in results.items():
                if isinstance(metrics, dict):
                    # Priority check for the keys in your snippet
                    if "acc,none" in metrics:
                        return metrics["acc,none"]
                    if "acc_norm,none" in metrics:
                        return metrics["acc_norm,none"]
                    if "acc" in metrics:
                        return metrics["acc"]

        # 2. Check top-level common keys (fallback)
        keys_to_check = ['eval_accuracy', 'accuracy', 'acc', 'val_accuracy', 'acc,none', 'acc_norm,none']
        for key in keys_to_check:
            if key in data:
                return data[key]
        
        # 3. Last resort: Recursively search for 'acc,none' or 'accuracy' in any nested dict
        # This is useful if the task name changes (not 'piqa')
        def recursive_search(d):
            for k, v in d.items():
                if k in keys_to_check:
                    return v
                if isinstance(v, dict):
                    result = recursive_search(v)
                    if result is not None:
                        return result
            return None

        if isinstance(data, dict):
            return recursive_search(data)

        return None
        
    except Exception as e:
        print(f"Error reading {os.path.basename(filepath)}: {e}")
        return None

def process_directory(dirpath):
    """
    Scans a directory for JSONs, extracts round numbers and accuracy.
    Returns: list of tuples [(round, accuracy), ...] sorted by round.
    """
    results = []
    
    try:
        files = os.listdir(dirpath)
    except FileNotFoundError:
        print(f"Error: Directory not found: {dirpath}")
        return []

    print(f"  > Scanning {len(files)} files in {os.path.basename(dirpath)}...")

    for filename in files:
        if not filename.endswith('.json'):
            continue
            
        round_num = get_round_from_filename(filename)
        
        if round_num is not None:
            full_path = os.path.join(dirpath, filename)
            acc = get_accuracy_from_json(full_path)
            
            if acc is not None:
                results.append((round_num, acc))
    
    # Sort by round number
    results.sort(key=lambda x: x[0])
    return results

def get_label_from_path(dirpath):
    """
    Generates a label based on the directory structure.
    Expected structure: .../fedavg_aggregation/fl/eval_results
    Goal Label: "fedavg_aggregation fl"
    """
    # Remove trailing slash if present
    dirpath = dirpath.rstrip('/')
    parts = dirpath.split('/')
    
    # Filter empty strings
    parts = [p for p in parts if p]
    
    if len(parts) >= 3:
        # If path ends in 'eval_results' (index -1), we want -3 and -2
        if parts[-1] == 'eval_results':
            return f"{parts[-3]} {parts[-2]}"
        # Fallback: just take the two parent folders
        return f"{parts[-2]} {parts[-1]}"
    
    # Fallback for short paths
    return os.path.basename(dirpath)

# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_accuracy.py /path/to/baseline1/eval_results /path/to/baseline2/eval_results ...")
        sys.exit(1)

    input_dirs = sys.argv[1:]
    
    # Create output directory
    output_dir = os.path.join('.', 'accuracy_graphs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    all_baselines_data = {}

    # --- Parse Data ---
    for dirpath in input_dirs:
        label = get_label_from_path(dirpath)
        print(f"Processing Baseline: {label}")
        
        data = process_directory(dirpath)
        
        if data:
            all_baselines_data[label] = data
        else:
            print(f"  > No valid data found for {label}")

    if not all_baselines_data:
        print("No valid data found to plot.")
        sys.exit(1)

    # --- Plotting ---
    print("\nGenerating Accuracy Plot...")
    plt.figure(figsize=(12, 7))

    for label, points in all_baselines_data.items():
        # Unzip list of tuples into two lists: rounds and accuracies
        rounds, accuracies = zip(*points)
        
        # Plot
        plt.plot(rounds, accuracies, marker='o', markersize=5, linewidth=2, label=label)

    plt.title("Comparison: Evaluation Accuracy per Round", fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save
    save_path = os.path.join(output_dir, "/files/comparison_graphs/compare_accuracy.png")
    plt.savefig(save_path)
    print(f"Done. Graph saved to: {save_path}")