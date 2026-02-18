import matplotlib.pyplot as plt
import sys
import ast
import os
import numpy as np
import json
import argparse

# ============================
# CONFIGURATION
# ============================
COLOR_MAPPING = {
    "FedAvg":       "#d62728",  # Red
    "FedMoECap-S": "#1f77b4",  # Blue
    "FedMoECap-R": "#ff7f0e",  # Orange
    "FedMoECap-P": "#2ca02c"   # Green
}

# Define the strict display order for the final table
DISPLAY_ORDER = ["FedAvg", "FedMoECap-S", "FedMoECap-P", "FedMoECap-R"]

def get_label_from_path(filepath):
    path_parts = filepath.strip().split('/')
    if 'fl_adaptive_freeze' in path_parts: return "FedMoECap-P"
    elif 'fl_swap' in path_parts: return "FedMoECap-R"
    elif 'fedavg' in path_parts: return "FedAvg"
    elif 'fl' in path_parts: return "FedMoECap-S"
    else: return os.path.basename(filepath).replace('.txt', '')

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

def plot_comparison(all_baselines, metric_key, title, ylabel, save_dir, filename, conversion_func=None):
    print(f"\n{'='*40}")
    print(f"METRIC: {title}")
    print(f"{'='*40}")
    
    plt.figure(figsize=(10, 6))
    has_data = False
    sorted_labels = sorted(all_baselines.keys())

    for label in sorted_labels:
        raw_data = all_baselines[label].get(metric_key)
        dtype = 'simple_list' if metric_key == 'server_data' else 'dict_list'
        
        # Plotting uses MEAN (Average per client) for readability
        y_values = extract_metric_per_round(raw_data, dtype, aggregation='mean')
        
        if not y_values: 
            print(f"  > {label}: [NO DATA]")
            continue

        has_data = True
        if conversion_func: y_values = [conversion_func(y) for y in y_values]
        
        # --- Print Data Points ---
        formatted_points = [round(y, 4) if isinstance(y, (float, int)) else y for y in y_values]
        print(f"  > {label} ({len(y_values)} rounds):")
        print(f"    {formatted_points}")
        
        rounds = range(1, len(y_values) + 1)
        color = COLOR_MAPPING.get(label, None)
        plt.plot(rounds, y_values, marker='o', markersize=4, label=label, linewidth=2, color=color)

    if not has_data:
        print(f"  > Skipping plot generation for {filename}")
        plt.close(); return

    plt.title(title, fontsize=14)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    print(f"\n  [Graph Saved]: {filename}")
    plt.close()

def load_client_batches(json_path, batch_size):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}"); return None

    batch_map = {}
    clients = data.get("clients", [])
    for cid, examples in enumerate(clients):
        num_batches = (len(examples) + batch_size - 1) // batch_size if batch_size > 0 else 0
        batch_map[cid] = num_batches
        batch_map[str(cid)] = num_batches
    return batch_map

def add_seconds_per_batch_metric(parsed_data, client_batches):
    if not parsed_data.get('training_time'): return
    raw_times = parsed_data['training_time']
    per_batch_list = []
    for round_data in raw_times:
        new_round_data = {}
        if isinstance(round_data, dict):
            for client_id, t_val in round_data.items():
                cid_key = str(client_id).split('-')[-1]
                if cid_key in client_batches and client_batches[cid_key] > 0:
                    new_round_data[client_id] = t_val / client_batches[cid_key]
                else:
                    new_round_data[client_id] = 0
        per_batch_list.append(new_round_data)
    parsed_data['seconds_per_batch'] = per_batch_list

# ============================
# STATISTICS
# ============================
def calculate_filtered_mean(values, label):
    if not values: return 0.0
    clean_values = [float(v) for v in values]
    if label == "FedMoECap-P":
        clean_values = [v for v in clean_values if v > 1e-4]
        if not clean_values: return 0.0
    return sum(clean_values) / len(clean_values)

def calculate_cumulative_sum(values):
    """Simple sum for total accumulation."""
    if not values: return 0.0
    return sum([float(v) for v in values])

def print_statistics(all_baselines):
    print(f"\n{'='*110}")
    print("FINAL STATISTICS: TOTALS & AVERAGES vs FedAvg")
    print(f"{'='*110}")

    # metric_key: (Display Name, Aggregation Type)
    # Aggregation Type: 'sum' for bandwidth, 'mean' for others
    metrics_map = {
        'total_bandwidth':   ('Total Bandwidth (MB)', 'sum'),
        'seconds_per_batch': ('Time/Batch (s)',      'mean'),
        'gpu_alloc':         ('Allocated VRAM (MB)',  'mean'),
        'training_time':     ('Total Train Time',     'sum') 
    }

    stats = {}
    
    for label, data in all_baselines.items():
        stats[label] = {}
        
        # --- 1. Bandwidth (Sum Server + Sum Client) ---
        srv = extract_metric_per_round(data.get('server_data'), 'simple_list')
        cli_sums = extract_metric_per_round(data.get('client_data'), 'dict_list', aggregation='sum')
        
        min_len = min(len(srv), len(cli_sums))
        bandwidth_per_round_mb = []
        if min_len > 0:
            bandwidth_per_round_mb = [(srv[i] + cli_sums[i]) / (1024**2) for i in range(min_len)]
        
        # Bandwidth is Cumulative (Sum)
        stats[label]['total_bandwidth'] = calculate_cumulative_sum(bandwidth_per_round_mb)
        
        # --- 2. Other Metrics ---
        # Seconds per Batch (Mean)
        if 'seconds_per_batch' in data:
             raw_vals = extract_metric_per_round(data['seconds_per_batch'], 'dict_list', aggregation='mean')
             stats[label]['seconds_per_batch'] = calculate_filtered_mean(raw_vals, label)
             
        # GPU Alloc (Mean)
        if 'gpu_alloc' in data:
             raw_vals = extract_metric_per_round(data['gpu_alloc'], 'dict_list', aggregation='mean')
             stats[label]['gpu_alloc'] = calculate_filtered_mean(raw_vals, label)

        # Training Time (Sum/Cumulative)
        if 'training_time' in data:
             # We take the average client time per round, then SUM those averages to get total training duration
             raw_vals = extract_metric_per_round(data['training_time'], 'dict_list', aggregation='mean')
             stats[label]['training_time'] = calculate_cumulative_sum(raw_vals)

    # Filter metrics to ensure they exist
    available_metrics = []
    for m_key in metrics_map:
        if any(m_key in stats[lbl] for lbl in stats):
            available_metrics.append(m_key)

    if not available_metrics: return

    # --- Print Table ---
    col_width = 24
    header_row = f"{'Baseline':<15}"
    sub_header_row = f"{'':<15}"
    
    for m in available_metrics:
        name, agg_type = metrics_map[m]
        header_row += f" | {name:<{col_width}}"
        sub_text = "(Total / vs FedAvg)" if agg_type == 'sum' else "(Avg   / vs FedAvg)"
        sub_header_row += f" | {sub_text:<{col_width}}"

    print("-" * len(header_row))
    print(header_row)
    print(sub_header_row)
    print("-" * len(header_row))

    # --- SORTING LOGIC ---
    # Filter only labels that exist in 'stats', then sort according to DISPLAY_ORDER
    existing_labels = [L for L in DISPLAY_ORDER if L in stats]
    
    # Add any labels that exist in data but were not in DISPLAY_ORDER (append to end)
    remaining_labels = sorted([L for L in stats.keys() if L not in DISPLAY_ORDER])
    final_sorted_labels = existing_labels + remaining_labels

    for label in final_sorted_labels:
        row_str = f"{label:<15}"
        
        for m in available_metrics:
            curr_val = stats[label].get(m, 0.0)
            
            # Comparison Logic
            comp_str = ""
            if label == "FedAvg":
                comp_str = "(ref)"
            elif "FedAvg" in stats:
                base_val = stats["FedAvg"].get(m, 0.0)
                if base_val > 0:
                    diff = ((curr_val - base_val) / base_val) * 100
                    sign = "+" if diff > 0 else ""
                    # PERCENTAGE: Max 1 decimal
                    comp_str = f"{sign}{diff:.1f}%"
                else:
                    comp_str = "N/A"
            else:
                comp_str = "-"

            # VALUE: Max 2 decimals
            val_str = f"{curr_val:.2f}"

            cell_content = f"{val_str} ({comp_str})"
            row_str += f" | {cell_content:<{col_width}}"
        
        print(row_str)
    
    print("-" * len(header_row))
    print("Note: Bandwidth & Training Time are CUMULATIVE (Sum of all rounds).")
    print("Note: VRAM & Seconds/Batch are AVERAGED per round.")


# ============================
# MAIN
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+')
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    output_dir = os.path.join('./files/', 'comparison_graphs')
    os.makedirs(output_dir, exist_ok=True)

    client_batches = None
    if args.json:
        print(f"Loading client batches from {args.json}...")
        client_batches = load_client_batches(args.json, args.batch_size)

    parsed_baselines = {}
    for filepath in args.files:
        label = get_label_from_path(filepath)
        print(f"Parsing: {label}...")
        parsed_data = parse_log_file(filepath)
        if parsed_data:
            if client_batches: 
                add_seconds_per_batch_metric(parsed_data, client_batches)
            parsed_baselines[label] = parsed_data

    if not parsed_baselines: sys.exit(1)

    print("\nGenerating Comparison Plots...")
    
    # Graphs
    plot_comparison(parsed_baselines, 'losses', "Comparison: Training Loss", "Loss", output_dir, "compare_losses.png")
    plot_comparison(parsed_baselines, 'trainable_experts', "Comparison: Active Experts", "Experts", output_dir, "compare_experts.png")
    
    if client_batches:
        plot_comparison(parsed_baselines, 'seconds_per_batch', "Comparison: Training Time (Seconds/Batch)", "Seconds per Batch", output_dir, "compare_time_per_batch.png")
    else:
        plot_comparison(parsed_baselines, 'training_time', "Comparison: Training Time", "Time (min)", output_dir, "compare_time.png")

    to_MB = lambda x: x / (1024.0**2)
    plot_comparison(parsed_baselines, 'server_data', "Comparison: Server Data", "Size (MB)", output_dir, "compare_server_data.png", conversion_func=to_MB)
    plot_comparison(parsed_baselines, 'client_data', "Comparison: Client Data", "Size (MB)", output_dir, "compare_client_data.png", conversion_func=to_MB)
    plot_comparison(parsed_baselines, 'gpu_alloc', "Comparison: GPU Allocated", "VRAM (MB)", output_dir, "compare_gpu_alloc.png")
    plot_comparison(parsed_baselines, 'gpu_res', "Comparison: GPU Reserved", "VRAM (MB)", output_dir, "compare_gpu_res.png")
    
    print_statistics(parsed_baselines)
    print("\nDone.")
