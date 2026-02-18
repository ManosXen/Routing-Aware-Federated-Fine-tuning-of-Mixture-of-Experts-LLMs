import matplotlib.pyplot as plt
import re
import argparse
import os
from datetime import datetime
import numpy as np

# ============================
# CONFIGURATION
# ============================

# 1. Define the specific order you want the bars to appear
ORDER_OF_BARS = ["FedAvg", "FedMoECap-S", "FedMoECap-R", "FedMoECap-P"]

# 2. Define Colors 
COLOR_MAPPING = {
    "FedAvg":      "#d62728",  # Red (Baseline)
    "FedMoECap-S": "#1f77b4",  # Blue (Standard FL)
    "FedMoECap-R": "#ff7f0e",  # Orange (Swap)
    "FedMoECap-P": "#2ca02c"   # Green (Proposed/Adaptive)
}

def get_label_from_path(filepath):
    """
    Splits the path into parts to safely identify the correct label,
    ignoring parent folders like 'fedavg_aggregation'.
    """
    # Normalize path separators and split into components
    # e.g., "/files/fedavg_aggregation/fl/stats/..." -> ['files', 'fedavg_aggregation', 'fl', 'stats', ...]
    path_parts = filepath.strip().split('/')
    
    # Check strict folder names
    if 'fl_adaptive_freeze' in path_parts:
        return "FedMoECap-P"
    elif 'fl_swap' in path_parts:
        return "FedMoECap-R"
    elif 'fedavg' in path_parts:
        return "FedAvg"
    elif 'fl' in path_parts:
        return "FedMoECap-S"
    else:
        # Fallback: simple filename check if folder structure isn't standard
        filename = os.path.basename(filepath)
        return filename

def parse_tegrastats_log(filepath):
    """Parses timestamps and power from log file."""
    timestamps = []
    power_watts = []
    
    # Regex for "mm-dd-yyyy HH:MM:SS ... VIN_SYS_5V0 <val>mW"
    log_pattern = re.compile(r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}).*?VIN_SYS_5V0\s+(\d+)mW')

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        matches = log_pattern.findall(content)
        for ts_str, p_mw in matches:
            try:
                dt_object = datetime.strptime(ts_str, "%m-%d-%Y %H:%M:%S")
                timestamps.append(dt_object)
                power_watts.append(float(p_mw) / 1000.0)
            except ValueError:
                continue
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

    if not timestamps: return None
    return {'timestamps': timestamps, 'power': power_watts}

def calculate_metrics(data):
    """Calculates Avg Power (W) and Total Energy (Wh)."""
    timestamps = data['timestamps']
    power = data['power']
    
    if len(timestamps) < 2: return None

    start_time = timestamps[0]
    time_seconds = [(t - start_time).total_seconds() for t in timestamps]
    
    # Integrate Power over Time to get Energy (Joules)
    if hasattr(np, 'trapezoid'):
        total_energy_joules = np.trapezoid(power, time_seconds)
    else:
        total_energy_joules = np.trapz(power, time_seconds)
    
    return {
        'avg_power_w': np.mean(power),
        'total_energy_wh': total_energy_joules / 3600.0
    }

def plot_bar_comparison(baselines_metrics, metric_key, title, ylabel, filename, save_dir, is_energy=False):
    plt.figure(figsize=(9, 6))
    
    # 1. Sort the data based on ORDER_OF_BARS
    # Filter out labels not present in the current data
    present_labels = [l for l in ORDER_OF_BARS if l in baselines_metrics]
    
    # Append any unexpected labels at the end
    extras = [l for l in baselines_metrics.keys() if l not in ORDER_OF_BARS]
    final_labels = present_labels + extras
    
    if not final_labels:
        print(f"Skipping {filename} (No matching labels found)")
        plt.close()
        return

    values = [baselines_metrics[l][metric_key] for l in final_labels]
    colors = [COLOR_MAPPING.get(l, '#7f7f7f') for l in final_labels]

    # 2. Create Bars
    bars = plt.bar(final_labels, values, color=colors, alpha=0.9, width=0.6)
    
    # 3. Calculate Savings (Relative to FedAvg if present)
    baseline_val = None
    if "FedAvg" in baselines_metrics:
        baseline_val = baselines_metrics["FedAvg"][metric_key]

    # 4. Annotate Bars
    for bar, label, val in zip(bars, final_labels, values):
        height = bar.get_height()
        
        # Display the absolute value on top
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.015),
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Display % reduction inside the bar (Only for Energy and if strictly better than FedAvg)
        if is_energy and baseline_val and label != "FedAvg" and val < baseline_val:
            saving = ((baseline_val - val) / baseline_val) * 100
            # Place text in the middle of the bar
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'-{saving:.1f}%',
                     ha='center', va='center', color='white', fontsize=12, fontweight='bold')

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    # Add 15% headroom for labels
    plt.ylim(top=max(values) * 1.15) 
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved {filename}")
    plt.close()

# ============================
# MAIN EXECUTION
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="List of tegrastats log files")
    args = parser.parse_args()

    output_dir = os.path.join('/files/', 'comparison_graphs')
    os.makedirs(output_dir, exist_ok=True)

    baselines_metrics = {}

    for filepath in args.files:
        # Determine the clean label using robust splitting
        label = get_label_from_path(filepath)
        print(f"Processing: {filepath} -> {label}")
        
        raw_data = parse_tegrastats_log(filepath)
        if raw_data:
            metrics = calculate_metrics(raw_data)
            if metrics:
                baselines_metrics[label] = metrics

    if not baselines_metrics:
        print("No valid data found.")
        exit(1)

    print("\nGenerating Final Plots...")

    # 1. Total Energy Plot
    plot_bar_comparison(
        baselines_metrics, 
        'total_energy_wh', 
        "Total Energy Consumption (Lower is Better)", 
        "Energy (Watt-hours)", 
        "final_energy_comparison.png", 
        output_dir, 
        is_energy=True
    )

    # 2. Average Power Plot
    plot_bar_comparison(
        baselines_metrics, 
        'avg_power_w', 
        "Average Power Draw", 
        "Power (Watts)", 
        "final_power_draw.png", 
        output_dir
    )
