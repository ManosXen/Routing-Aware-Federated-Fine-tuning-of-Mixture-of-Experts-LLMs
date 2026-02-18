#!/usr/bin/env python3
import sys
import re
import torch
from partial_aggr_job import PartialAggrJob
import os

from client_trainer import PartialClientTrainer 

import re

def parse_config(file_path):
    job_config = {}
    client_configs = {}
    
    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # 1. Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # 2. Parse Client Lines
        # Regex matches "client 1:", "Client 2 :", etc.
        client_match = re.match(r"client\s+(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if client_match:
            client_id = int(client_match.group(1))
            params = [p.strip() for p in client_match.group(2).split(",")]

            if len(params) < 8:
                raise ValueError(f"Client {client_id} requires at least 8 parameters. Found {len(params)}.")

            client_configs[f"site-{client_id}"] = {
                "freeze_criterion": params[0].lower(),
                "freeze_value":     float(params[1]) if params[1] else 0.0,
                "qb":               int(params[2]),
                "lr":               float(params[3]),
                "epochs":           int(params[4]),
                "batch_size":       int(params[5]),
                "dt":               params[6],
                "dataset_file":     params[7],
            }
            continue

        # 3. Parse Global Variables (e.g., n_clients = 5)
        if "=" in line:
            key, val = [v.strip() for v in line.split("=", 1)]
            # Convert to numeric if possible, else keep as string
            if val.replace('.', '', 1).isdigit():
                val = float(val) if '.' in val else int(val)
            job_config[key.lower()] = val

    return job_config, client_configs

# Usage
# config, clients = parse_config("config.txt")

def compute_dataset_ranges(size_list):
    ranges = []
    start = 0
    for sz in size_list:
        if sz in (-1, None):
            ranges.append((-1, -1))
        else:
            end = start + sz
            ranges.append((start, end))
            start = end
    return ranges


def main():
    if len(sys.argv) < 3:
        print("Usage: python fl_script_from_config.py <path_to_config.txt> n_threads")
        sys.exit(1)

    config_path = sys.argv[1]
    n_threads=int(sys.argv[2]) # ensure n_threads is int
    print(f"Loading configuration from: {config_path}")

    save_file = config_path.split('/')[-1]

    job_config, client_configs = parse_config(config_path)

    n_clients = int(job_config.get("n_clients", len(client_configs)))
    num_rounds = int(job_config.get("num_rounds", 5))
    rank = int(job_config.get("rank", 0))

    threshold = float(job_config.get("threshold", 0.1))
    patience = int(job_config.get("patience", 1))
    
    # --- NEW: Read hard convergence threshold (default to 0.6 if missing) ---
    hard_convergence = float(job_config.get("hard_convergence_threshold", 0.6))

    print(f"Parsed {n_clients} clients, {num_rounds} rounds")
    print(f"Adaptive freeze params: threshold={threshold}, patience={patience}, hard_convergence={hard_convergence}")

    # Create federated job
    # --- NEW: Pass hard_convergence to the Job ---
    job = PartialAggrJob(
        n_clients=n_clients,
        num_rounds=num_rounds,
        threshold=threshold,
        patience=patience,
        hard_convergence=hard_convergence
    )

    # Add clients dynamically
    for idx, (site, params) in enumerate(sorted(client_configs.items(), key=lambda x: int(x[0].split('-')[1]))):
        
        # Collect all args for the Executor
        client_args = {
            "fc": params["freeze_criterion"],
            "fr_or_thr": params["freeze_value"],
            "qb": params["qb"],
            "lr": params.get('lr', 1e-5),
            "epochs": params.get('epochs', 1),
            "dt": params.get("dt", None),
            "dataset_file": params.get("dataset_file", None),
            "batch_size": params.get('batch_size', 8),
            "rank": rank,
            "save_file_aux": save_file,
        }

        # Instantiate the Executor with its args
        executor = PartialClientTrainer(**client_args)

        # Assign the Executor instance to the site
        job.to(executor, site)

        print(f"Added {site} with Executor args: {client_args}")

    job.export_job("/tmp/nvflare/jobs")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_ids = ",".join(str(i) for i in range(n_gpus))
        print(f"Running simulator on {n_gpus} available GPUs: {gpu_ids}")
    else:
        print("Warning: No GPUs detected. Running on CPU.")
        gpu_ids = "" 

    if n_threads == -1:
        job.simulator_run("/tmp/nvflare/jobs/workdir", gpu=gpu_ids)
    else:
        job.simulator_run("/tmp/nvflare/jobs/workdir", gpu=gpu_ids, threads=n_threads)


if __name__ == "__main__":
    main()