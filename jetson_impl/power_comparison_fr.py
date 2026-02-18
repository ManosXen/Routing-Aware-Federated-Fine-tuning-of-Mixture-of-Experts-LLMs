import matplotlib.pyplot as plt
import re
import argparse
import os
from datetime import datetime
import numpy as np

# ============================
# CONFIGURATION
# ============================

def get_label_from_path(filepath):
    """
    Extracts the freeze rate from the filename using the logic from your training script.
    Example input: fl/stats/power_boolq_03_act_095_thr_015.txt
    """
    filename = os.path.basename(filepath).lower()
    
    # 1. Handle FedAvg case (Frozen is effectively 0)
    if "fedavg" in filename:
        return "Freeze Rate: 0"

    # 2. Regex to find 'act_095', 'act_08', etc.
    match = re.search(r'_act_(\d+)_', filename)
    
    if match:
        rate_str = match.group(1)
        
        # Remove leading zero if it exists (e.g., "08" -> "8")
        # unless it is literally "0"
        if rate_str.startswith('0') and len(rate_str) > 1:
            rate_str = rate_str.lstrip('0')
        
        # Construct float string "0.XXX"
        try:
            val = float(f"0.{rate_str}")
            return f"Freeze Rate: {val}"
        except ValueError:
            return f"Freeze Rate: 0.{rate_str}"
            
    else:
        # 3. Default fallback
        return "Freeze Rate: 0.9"

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
    # Handle numpy version compatibility for trapz/trapezoid
    if hasattr(np, 'trapezoid'):
        total_energy_joules = np.trapezoid(power, time_seconds)
    else:
        total_energy_joules = np.trapz(power, time_seconds)
    
    return {
        'avg_power_w': np.mean(power),
        'total_energy_wh': total_energy_joules / 3600.0
    }

def plot_bar_comparison(baselines_metrics, metric_key, title, ylabel, filename, save_dir, is_energy=False):
    plt.figure(figsize=(10, 6))
    
    # 1. Sort labels numerically based on the freeze rate value
    # Function to extract float from "Freeze Rate: 0.95"
    def sort_key(label):
        try:
            return float(label.split(': ')[1])
        except:
            return 0.0

    sorted_labels = sorted(baselines_metrics.keys(), key=sort_key)
    
    if not sorted_labels:
        print(f"Skipping {filename} (No labels found)")
        plt.close()
        return

    values = [baselines_metrics[l][metric_key] for l in sorted_labels]
    
    # 2. Dynamic Colors (Tab10)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(sorted_labels))]

    # 3. Create Bars
    bars = plt.bar(sorted_labels, values, color=colors, alpha=0.9, width=0.6)
    
    # 4. Calculate Savings (Relative to Freeze Rate: 0 / FedAvg if present)
    baseline_val = None
    if "Freeze Rate: 0" in baselines_metrics:
        baseline_val = baselines_metrics["Freeze Rate: 0"][metric_key]
    elif "Freeze Rate: 0.0" in baselines_metrics:
        baseline_val = baselines_metrics["Freeze Rate: 0.0"][metric_key]

    # 5. Annotate Bars
    for bar, label, val in zip(bars, sorted_labels, values):
        height = bar.get_height()
        
        # Display the absolute value on top
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(values)*0.015),
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Display % reduction inside the bar (Only if strictly better than Baseline 0)
        if is_energy and baseline_val and (val < baseline_val) and ("Freeze Rate: 0" not in label):
            saving = ((baseline_val - val) / baseline_val) * 100
            # Place text in the middle of the bar
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'-{saving:.1f}%',
                     ha='center', va='center', color='white', fontsize=11, fontweight='bold')

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45) # Rotate labels slightly if they get long
    
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

    # Output directory
    output_dir = os.path.join('/files/', 'comparison_graphs')
    os.makedirs(output_dir, exist_ok=True)

    baselines_metrics = {}

    for filepath in args.files:
        # Determine the label based on freeze rate
        label = get_label_from_path(filepath)
        print(f"Processing: {os.path.basename(filepath)} -> {label}")
        
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
        "Total Energy Consumption by Freeze Rate", 
        "Energy (Watt-hours)", 
        "compare_energy_freeze.png", 
        output_dir, 
        is_energy=True
    )

    # 2. Average Power Plot
    plot_bar_comparison(
        baselines_metrics, 
        'avg_power_w', 
        "Average Power Draw by Freeze Rate", 
        "Power (Watts)", 
        "compare_power_freeze.png", 
        output_dir
    )
