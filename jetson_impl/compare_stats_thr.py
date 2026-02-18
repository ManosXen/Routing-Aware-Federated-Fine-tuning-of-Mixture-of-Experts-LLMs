import matplotlib.pyplot as plt
import sys
import ast
import os
import numpy as np
import json
import argparse
import re

# ============================
# CONFIGURATION
# ============================

def get_label_from_path(filepath):
    """
    Extracts the threshold from the filename.
    1. If filename contains 'fedavg' -> Threshold: 0
    2. If filename contains '_act_thr_XXX' -> Extract threshold (e.g. 010 -> 0.10)
    3. Else -> Threshold: 0.15 (Default)
    """
    filename = os.path.basename(filepath).lower()
    
    # 1. Handle FedAvg case
    if "fedavg" in filename:
        return "Threshold: 0"

    # 2. Regex to find 'act_thr_010', 'act_thr_05', etc.
    match = re.search(r'_act_thr_(\d+)', filename)
    
    if match:
        thr_str = match.group(1)
        # Remove leading zero if it exists (e.g., "010" -> "10")
        if thr_str.startswith('0') and len(thr_str) > 1:
            thr_str = thr_str.lstrip('0')
        
        # Construct float string "0.XXX"
        try:
            val = float(f"0.{thr_str}")
            return f"Threshold: {val}"
        except ValueError:
            return f"Threshold: 0.{thr_str}"
            
    else:
        # 3. Default fallback
        return "Threshold: 0.15"

def parse_log_file(filepath):
    data = {'losses': [], 'server_data': [], 'client_data': [], 'training_time': [], 
            'trainable_experts': [], 'gpu_alloc': [], 'gpu_res': []}
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            def get_next():
                if i + 1 < len(lines):
                    try: return ast.literal_eval(lines[i+1].strip())
                    except: return None
                return None
            
            if line == "Losses:": data['losses'] = get_next()
            elif line == "Size of data sent from the server:": data['server_data'] = get_next()
            elif line == "Size of data sent from the clients:": data['client_data'] = get_next()
            elif line == "Client training time:": data['training_time'] = get_next()
            elif line == "Trainable Experts History:": data['trainable_experts'] = get_next()
            elif line == "Allocated GPU VRAM:": data['gpu_alloc'] = get_next()
            elif line == "Reserved GPU VRAM:": data['gpu_res'] = get_next()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    return data

# --- UPDATED: Helper to handle SUM vs MEAN aggregation ---
def extract_metric_per_round(metric_data, data_type='dict_list', aggregation='mean'):
    """
    Extracts a list of values per round.
    aggregation: 'mean' (average of clients) or 'sum' (total of clients)
    """
    if not metric_data: return []
    result = []
    
    if data_type == 'simple_list': return metric_data
    
    for round_data in metric_data:
        if isinstance(round_data, dict) and round_data:
            vals = list(round_data.values())
            if aggregation == 'sum':
                result.append(sum(vals))
            else:
                result.append(sum(vals) / len(vals))
        elif isinstance(round_data, list):
            total, count = 0, 0
            for item in round_data:
                if isinstance(item, dict) and 'rec' in item:
                     total += item['rec']; count += 1
            if aggregation == 'sum':
                result.append(total)
            else:
                result.append(total / count if count > 0 else 0)
        else: 
            result.append(0)
    return result

def extract_mean_per_round(metric_data, data_type='dict_list'):
    """Backwards compatibility wrapper for plotting function"""
    return extract_metric_per_round(metric_data, data_type, aggregation='mean')

def plot_comparison(all_baselines, metric_key, title, ylabel, save_dir, filename, conversion_func=None):
    plt.figure(figsize=(10, 6))
    has_data = False
    
    # Sort labels to ensure legend order is logical (e.g., 0, 0.1, 0.15)
    def sort_key(label):
        try:
            return float(label.split(': ')[1])
        except:
            return 0.0

    sorted_labels = sorted(all_baselines.keys(), key=sort_key)
    
    # --- COLOR FIX ---
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_labels))]

    for idx, label in enumerate(sorted_labels):
        raw_data = all_baselines[label].get(metric_key)
        dtype = 'simple_list' if metric_key == 'server_data' else 'dict_list'
        y_values = extract_mean_per_round(raw_data, dtype)
        
        if not y_values: continue
        has_data = True
        if conversion_func: y_values = [conversion_func(y) for y in y_values]
        
        rounds = range(1, len(y_values) + 1)
        plt.plot(rounds, y_values, marker='o', markersize=4, label=label, 
                 linewidth=2, color=colors[idx % len(colors)])

    if not has_data:
        print(f"  > Skipping {filename} (No data)")
        plt.close(); return

    plt.title(title, fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    print(f"  > Saved {filename}")
    plt.close()

def load_client_batches(json_path, batch_size):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

    batch_map = {}
    clients = data.get("clients", [])
    
    for cid, examples in enumerate(clients):
        num_batches = (len(examples) + batch_size - 1) // batch_size if batch_size > 0 else 0
        batch_map[cid] = num_batches
        batch_map[str(cid)] = num_batches
        
    print(batch_map)
    return batch_map

def add_seconds_per_batch_metric(parsed_data, client_batches):
    if not parsed_data.get('training_time'): return

    raw_times = parsed_data['training_time']
    per_batch_list = []
    
    for round_data in raw_times:
        new_round_data = {}
        for cl in round_data.items():
            cid_key = cl[0].split('-')[-1]
            if cid_key in client_batches and client_batches[cid_key] > 0:
                t = cl[1] / client_batches[cid_key]
                new_round_data[cl[0]] = t
            else:
                new_round_data[cl[0]] = 0

        per_batch_list.append(new_round_data)

    parsed_data['seconds_per_batch'] = per_batch_list

# ============================
# STATISTICS TABLE
# ============================
def calculate_filtered_mean(values):
    if not values: return 0.0
    clean_values = [float(v) for v in values]
    clean_values = [v for v in clean_values if v > 1e-4]
    if not clean_values: return 0.0
    return sum(clean_values) / len(clean_values)

def calculate_cumulative_sum(values):
    if not values: return 0.0
    return sum([float(v) for v in values])

def print_statistics(all_baselines):
    # Set the target reference for comparisons
    TARGET_REF = "Threshold: 0.15"

    print(f"\n{'='*110}")
    print(f"FINAL STATISTICS: THRESHOLD COMPARISON (vs {TARGET_REF})")
    print(f"{'='*110}")

    metrics_map = {
        'total_bandwidth':   ('Total Bandwidth (MB)', 'sum'),
        'seconds_per_batch': ('Time/Batch (s)',      'mean'),
        'gpu_alloc':         ('Allocated VRAM (MB)',  'mean'),
        'training_time':     ('Total Train Time',     'sum') 
    }

    stats = {}
    
    for label, data in all_baselines.items():
        stats[label] = {}
        
        # Bandwidth
        srv = extract_metric_per_round(data.get('server_data'), 'simple_list')
        cli_sums = extract_metric_per_round(data.get('client_data'), 'dict_list', aggregation='sum')
        
        min_len = min(len(srv), len(cli_sums))
        bandwidth_per_round_mb = []
        if min_len > 0:
            bandwidth_per_round_mb = [(srv[i] + cli_sums[i]) / (1024**2) for i in range(min_len)]
        
        stats[label]['total_bandwidth'] = calculate_cumulative_sum(bandwidth_per_round_mb)
        
        # Other Metrics
        if 'seconds_per_batch' in data:
             raw_vals = extract_metric_per_round(data['seconds_per_batch'], 'dict_list', aggregation='mean')
             stats[label]['seconds_per_batch'] = calculate_filtered_mean(raw_vals)
        if 'gpu_alloc' in data:
             raw_vals = extract_metric_per_round(data['gpu_alloc'], 'dict_list', aggregation='mean')
             stats[label]['gpu_alloc'] = calculate_filtered_mean(raw_vals)
        if 'training_time' in data:
             raw_vals = extract_metric_per_round(data['training_time'], 'dict_list', aggregation='mean')
             stats[label]['training_time'] = calculate_cumulative_sum(raw_vals)

    # Filter metrics
    available_metrics = []
    for m_key in metrics_map:
        if any(m_key in stats[lbl] for lbl in stats):
            available_metrics.append(m_key)
    if not available_metrics: return

    # Sorting Logic (Numerical by Threshold)
    def sort_key(label):
        try: return float(label.split(': ')[1])
        except: return -1.0
    sorted_labels = sorted(stats.keys(), key=sort_key)
    
    # Identify Reference
    ref_label = None
    if TARGET_REF in stats:
        ref_label = TARGET_REF
    else:
        print(f"[!] Warning: '{TARGET_REF}' not found. Using first available.")
        ref_label = sorted_labels[0] if sorted_labels else None

    # Print Header
    col_width = 24
    header_row = f"{'Threshold':<15}"
    sub_header_row = f"{'':<15}"
    
    for m in available_metrics:
        name, agg_type = metrics_map[m]
        header_row += f" | {name:<{col_width}}"
        sub_text = "(Total / vs Ref)" if agg_type == 'sum' else "(Avg   / vs Ref)"
        sub_header_row += f" | {sub_text:<{col_width}}"

    print("-" * len(header_row))
    print(header_row)
    print(sub_header_row)
    print("-" * len(header_row))

    for label in sorted_labels:
        row_str = f"{label:<15}"
        
        for m in available_metrics:
            curr_val = stats[label].get(m, 0.0)
            
            comp_str = ""
            if label == ref_label:
                comp_str = "(ref)"
            elif ref_label and ref_label in stats:
                base_val = stats[ref_label].get(m, 0.0)
                if base_val > 0:
                    diff = ((curr_val - base_val) / base_val) * 100
                    sign = "+" if diff > 0 else ""
                    comp_str = f"{sign}{diff:.1f}%"
                else:
                    comp_str = "N/A"
            else:
                comp_str = "-"

            # Formatting
            val_str = f"{curr_val:.2f}"
            cell_content = f"{val_str} ({comp_str})"
            row_str += f" | {cell_content:<{col_width}}"
        
        print(row_str)
    
    print("-" * len(header_row))
    print(f"Reference Baseline: {ref_label}")
    print("Note: Bandwidth & Training Time are CUMULATIVE.")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+')
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    output_dir = os.path.join('/files/', 'comparison_graphs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    client_batches = None
    if args.json:
        print(f"Loading client batches from {args.json}...")
        client_batches = load_client_batches(args.json, args.batch_size)

    parsed_baselines = {}
    for filepath in args.files:
        label = get_label_from_path(filepath)
        print(f"Parsing: {os.path.basename(filepath)} -> {label}")
        
        parsed_data = parse_log_file(filepath)
        if parsed_data:
            if client_batches: 
                add_seconds_per_batch_metric(parsed_data, client_batches)
            parsed_baselines[label] = parsed_data

    if not parsed_baselines: sys.exit(1)

    print("\nGenerating Comparison Plots...")
    
    plot_comparison(parsed_baselines, 'losses', "Comparison: Training Loss by Threshold", "Loss", output_dir, "compare_losses_threshold.png")
    plot_comparison(parsed_baselines, 'trainable_experts', "Comparison: Active Experts by Threshold", "Experts", output_dir, "compare_experts_threshold.png")
    
    if client_batches:
        plot_comparison(parsed_baselines, 'seconds_per_batch', "Comparison: Training Time (Seconds/Batch)", "Seconds per Batch", output_dir, "compare_time_per_batch_threshold.png")
    else:
        plot_comparison(parsed_baselines, 'training_time', "Comparison: Training Time", "Time (min)", output_dir, "compare_time_threshold.png")

    to_MB = lambda x: x / (1024.0**2)
    plot_comparison(parsed_baselines, 'server_data', "Comparison: Server Data", "Size (MB)", output_dir, "compare_server_data_threshold.png", conversion_func=to_MB)
    plot_comparison(parsed_baselines, 'client_data', "Comparison: Client Data", "Size (MB)", output_dir, "compare_client_data_threshold.png", conversion_func=to_MB)
    plot_comparison(parsed_baselines, 'gpu_alloc', "Comparison: GPU Allocated", "VRAM (MB)", output_dir, "compare_gpu_alloc_threshold.png")
    plot_comparison(parsed_baselines, 'gpu_res', "Comparison: GPU Reserved", "VRAM (MB)", output_dir, "compare_gpu_res_threshold.png")

    # --- Print Stats ---
    print_statistics(parsed_baselines)

    print("\nDone.")
