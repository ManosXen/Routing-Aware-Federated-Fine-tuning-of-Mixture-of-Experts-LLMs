import sys
import os
import subprocess
import shutil

# --- NEW: Check for override flag ---
force_merge = False
if "--force" in sys.argv:
    force_merge = True
    sys.argv.remove("--force") # Remove from args so it doesn't mess up indices below

if len(sys.argv) < 6:
    print("Usage: python script.py <folder> <task> <task_short> <qb> <rank> [--force]")
    sys.exit(1)

fl_folder = sys.argv[1]
task = sys.argv[2]
task_short = sys.argv[3]
qb = sys.argv[4]
rank = sys.argv[5]

# Safety check for path parsing
if len(fl_folder.split('/')) < 4:
     print("Error: fl_folder path structure does not match expected depth (aggr/method).")
     # You might want to handle this differently, but keeping generic for now
     aggr_strategy = "unknown" 
     fl_method = "unknown"
else:
    aggr_strategy = fl_folder.split('/')[2]
    fl_method = fl_folder.split('/')[3]

stats_folder = os.path.join(fl_folder, 'stats')
adapter_temp_folder = os.path.join(fl_folder, 'adapter_temp')
eval_results_folder = os.path.join(fl_folder, 'eval_results')

os.makedirs(adapter_temp_folder, exist_ok=True)
os.makedirs(eval_results_folder, exist_ok=True)

# --- MERGE LOOP ---
for file in os.listdir(stats_folder):
    if not file.endswith(".pt") or task_short not in file:
        continue
    
    # Use os.path.join for better cross-platform compatibility
    filename_no_ext = os.path.splitext(file)[0]
    adapter_save_folder = os.path.join(adapter_temp_folder, filename_no_ext)
    
    # --- NEW: Check if merge is done ---
    # Check if folder exists AND has contents (is not empty)
    is_merged = os.path.exists(adapter_save_folder) and len(os.listdir(adapter_save_folder)) > 0
    
    if is_merged:
        if force_merge:
            print(f"Merge exists for {file}, but --force used. Overwriting...")
        else:
            print(f"Merge already done for {file}. Skipping...")
            continue
    
    # Ensure folder is created (idempotent)
    os.makedirs(adapter_save_folder, exist_ok=True)

    adapter_file_path = os.path.join(stats_folder, file)
    
    cmd = [
        "python3", "/files/jetson_impl/merge_model_adapter.py",
        "--adapter_file", adapter_file_path,
        "--save_path", adapter_save_folder,
        "--qb", qb,
        "--rank", rank
    ]
    
    print(f"Running merge for {file}...")
    subprocess.run(cmd, check=True)

# --- EVALUATION LOOP ---
os.makedirs(os.path.join(fl_folder, 'eval_results'), exist_ok=True)

for file in os.listdir(adapter_temp_folder):
        if task_short in file:
            filename = os.path.splitext(file)[0]
            
            output_path = os.path.join(eval_results_folder, f"{filename}.json")
            adapter_path = os.path.join(adapter_temp_folder, file)
            
            # Check if adapter path is actually a directory (since merge creates a folder)
            # If your merge script creates a single file, remove the check below, 
            # but usually adapters are saved as folders containing .bin and .json
            if not os.path.isdir(adapter_path):
                 # Handle case where 'file' might be a remnant file rather than a folder
                 continue

            model_args = f"pretrained='allenai/OLMoE-1B-7B-0125',load_in_8bit=True,peft={adapter_path}"
            
            cmd = [
                "lm_eval",
                "--model", "hf",
                "--model_args", model_args,
                "--task", task,
                "--device", "cuda:0",
                "--batch_size", "auto",
                "--output_path", output_path
            ]
            
            print(f"Evaluating {filename}...")
            # Added check=False so one eval failure doesn't crash the whole loop
            subprocess.run(cmd, check=False)
