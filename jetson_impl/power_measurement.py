import sys
import subprocess
import time
import os
import re

def get_container_command(container_identifier):
    """
    Retrieves the full command string used to start the container.
    """
    try:
        # Get the command array from docker inspect (e.g., ["python3", "/files/...", ...])
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{range .Config.Cmd}}{{.}} {{end}}", container_identifier],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"Error inspecting container: {e}")
        return None

def is_container_active(container_identifier):
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_identifier],
            capture_output=True, text=True
        )
        return result.stdout.strip() == 'true'
    except:
        return False

# --- CONFIGURATION ---
if len(sys.argv) < 2:
    print("Usage: sudo python3 power_measurement.py <container_id>")
    sys.exit(1)

CONTAINER_ID = sys.argv[1]
CHECK_INTERVAL = 5

# 1. Validation
if not is_container_active(CONTAINER_ID):
    print(f"Error: Container '{CONTAINER_ID}' is not running.")
    sys.exit(1)

# 2. Extract Command to determine Filename
full_cmd = get_container_command(CONTAINER_ID)
print(f"Detected Command: {full_cmd}")

# Logic to find the config file (.txt) in the command
# We look for the argument that ends in .txt
match = re.search(r'(\S+\.txt)', full_cmd)

if match:
    # This captures the path inside docker: /files/.../boolq_03_act_thr_015.txt
    docker_config_path = match.group(1)
    
    # Extract just the filename: boolq_03_act_thr_015.txt
    config_basename = os.path.basename(docker_config_path)
    
    # Extract the script directory inside docker to map it to host
    # Assuming the structure: python3 /files/path/to/script.py
    # We find the python script argument to locate the folder
    script_match = re.search(r'(\S+\.py)', full_cmd)
    if script_match:
        docker_script_path = script_match.group(1)
        docker_dir = os.path.dirname(docker_script_path)
        
        # MAPPING: Convert Docker path (/files/...) to Host path (/home/mxenos/...)
        # You can adjust this replacement if your mapping is different
        host_dir = docker_dir.replace('/files', '/home/mxenos')
        
        # Construct final paths
        stats_dir = os.path.join(host_dir, "stats")
        log_filename = f"power_{os.path.splitext(config_basename)[0]}.txt"
        LOG_FILE = os.path.join(stats_dir, log_filename)
    else:
        print("Error: Could not find a .py script in the command to determine directory.")
        sys.exit(1)
else:
    print("Error: Could not find a .txt config file in the container command.")
    sys.exit(1)

# Ensure stats directory exists
os.makedirs(stats_dir, exist_ok=True)

print(f"--- Power Monitor (Auto-Name) ---")
print(f"Target Container: {CONTAINER_ID}")
print(f"Log File:         {LOG_FILE}")
print(f"---------------------------------")

# 3. Start Logging
log_f = open(LOG_FILE, "w")
tegra_process = subprocess.Popen(
    ["tegrastats", "--interval", "3000"], 
    stdout=log_f, 
    stderr=subprocess.PIPE
)

try:
    print("Monitoring container... (Press Ctrl+C to stop manually)")
    start_time = time.time()
    
    while True:
        if not is_container_active(CONTAINER_ID):
            print(f"\nContainer stopped. Stopping logger.")
            break
        
        elapsed = int(time.time() - start_time)
        sys.stdout.write(f"\rStatus: Running | Elapsed: {elapsed}s | Log: {log_filename}")
        sys.stdout.flush()
        time.sleep(CHECK_INTERVAL)

except KeyboardInterrupt:
    print("\nManual stop detected.")

finally:
    if 'tegra_process' in locals():
        tegra_process.terminate()
        try:
            tegra_process.wait(timeout=2)
        except:
            tegra_process.kill()
    if 'log_f' in locals():
        log_f.close()
    print(f"\nDone.")
