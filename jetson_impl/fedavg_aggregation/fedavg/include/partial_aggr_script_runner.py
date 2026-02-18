#!/usr/bin/env python3
import sys
import re
import torch
from nvflare.job_config.script_runner import ScriptRunner
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from partial_aggr_job import PartialAggrJob
import os


def parse_config(file_path):
    """
    Parse the text-based config into job settings and client configs.

    Supported global keys:
      - n_clients = int
      - num_rounds = int
      - qb = int
      - rank = int
      - dataset = <huggingface-id>
      - split = val1, val2, ...
      - size = val1, val2, ...
    Client lines:
      client <id>: freeze_criterion, freeze_value, lr, epochs, batch_size
    """

    with open(file_path, "r") as f:
        raw_lines = f.readlines()

    # sanitize: strip trailing newline but preserve start-of-line to detect comments
    lines = []
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        lines.append(stripped)

    job_config = {}
    client_configs = {}

    # temporary lists for per-client split/size if provided as comma lists
    split_list = []
    size_list = []

    for line in lines:
        # client lines
        m = re.match(r"client\s+(\d+)\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            client_id = int(m.group(1))
            params = [p.strip() for p in m.group(2).split(",")]
            # expected 5 params: freeze_criterion, freeze_value, lr, epochs, batch_size
            if len(params) < 5:
                raise ValueError(f"Client {client_id} line must have 5 comma-separated values. Got: {params}")

            freeze_criterion = params[0].lower()
            freeze_value = float(params[1]) if params[1] != "" else 0.0
            lr = float(params[2]) if params[2] != "" else None
            epochs = int(params[3]) if params[3] != "" else None
            batch_size = int(params[4]) if params[4] != "" else None

            client_configs[f"site-{client_id}"] = {
                "freeze_criterion": freeze_criterion,
                "freeze_value": freeze_value,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                # placeholders filled later from global dataset/split/size lists:
                "dataset": None,
                "dataset_split": None,
                "range": None,
            }
        else:
            # global key = val
            if "=" not in line:
                # ignore weird lines
                continue
            key, val = [v.strip() for v in line.split("=", 1)]
            kl = key.lower()

            # special handling for list fields
            if kl == "split":
                # split = none, none  -> list of strings
                split_list = [s.strip() for s in val.split(",")]
                job_config["split_list"] = split_list
                continue
            if kl == "size":
                # size = 10000, max -> convert "max" -> -1, ints -> int
                raw_sizes = [s.strip() for s in val.split(",")]
                parsed = []
                for s in raw_sizes:
                    if s.lower() in ("max", "none", ""):
                        parsed.append(-1)
                    else:
                        try:
                            parsed.append(int(s))
                        except Exception:
                            parsed.append(-1)
                size_list = parsed
                job_config["size_list"] = size_list
                continue

            # store dataset as raw string
            if kl == "dataset":
                job_config["dataset"] = val
                continue

            # try to eval simple python literals (int, float, True/False)
            try:
                job_config[kl] = eval(val)
            except Exception:
                job_config[kl] = val

    # Fill per-client dataset, split and size using available lists (by client index)
    # If split_list or size_list shorter than number of clients, use defaults
    n_clients_provided = len(client_configs)
    # use dataset from job_config if present
    dataset_global = job_config.get("dataset", None)
    split_list = job_config.get("split_list", [])
    size_list = job_config.get("size_list", [])

    # If user provided explicit n_clients global value, respect it (but we still only have as many client lines as given)
    for idx, (site, cfg) in enumerate(sorted(client_configs.items(), key=lambda x: int(x[0].split("-")[1]))):
        # index into lists is idx (0-based)
        cfg["dataset"] = dataset_global
        if idx < len(split_list):
            cfg["dataset_split"] = split_list[idx]
        else:
            cfg["dataset_split"] = "none"
        if idx < len(size_list):
            cfg["range"] = size_list[idx]
        else:
            cfg["range"] = -1

        client_configs[site] = cfg

    return job_config, client_configs


def build_model(qb, rank):
    """Load and prepare the model with quantization options."""
    base_model = "allenai/OLMoE-1B-7B-0125"

    if qb == 4:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif qb == 8:
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    if qconfig:
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=qconfig)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model).to("cuda")

    if rank!=0:
        config = LoraConfig(
            r=rank,
            lora_alpha=rank*2,
            target_modules=["gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config).to("cuda")

    return model

def compute_dataset_ranges(size_list):
    """Compute start/end index pairs for homogeneous FL from a list of sizes."""
    ranges = []
    start = 0
    for sz in size_list:
        if sz in (-1, None):  # -1 or None means use all -> skip homogeneous slicing
            ranges.append((-1, -1))
        else:
            end = start + sz
            ranges.append((start, end))
            start = end
    return ranges


def main():
    if len(sys.argv) < 2:
        print("Usage: python fl_script_from_config.py <path_to_config.txt>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading configuration from: {config_path}")

    job_config, client_configs = parse_config(config_path)

    n_clients = int(job_config.get("n_clients", len(client_configs)))
    num_rounds = int(job_config.get("num_rounds", 5))
    qb = int(job_config.get("qb", 8))
    rank = int(job_config.get("rank", 0))
    dataset_global = job_config.get("dataset", "unknown")

    print(f"Parsed {n_clients} clients, {num_rounds} rounds, quantization: {qb}-bit")

    model = build_model(qb, rank)

    # --- Detect FL mode ---
    size_list = job_config.get("size_list", [])
    hetero_mode = any(sz == -1 for sz in size_list) or len(set(job_config.get("split_list", []))) > 1
    if hetero_mode:
        print("Mode: HETEROGENEOUS FL")
    else:
        print("Mode: HOMOGENEOUS FL")

    # Pre-compute start/end indices if homogeneous
    if not hetero_mode:
        ranges = compute_dataset_ranges(size_list)
    else:
        ranges = [(-1, -1)] * n_clients

    # Create federated job
    job = PartialAggrJob(
        name="test-partial-avg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=model,
    )

    train_script = "main_v2_fl.py"

    # Add clients dynamically
    for idx, (site, params) in enumerate(sorted(client_configs.items(), key=lambda x: int(x[0].split('-')[1]))):
        fc = params["freeze_criterion"]
        fr_or_thr = params["freeze_value"]
        dt = params.get("dataset", dataset_global)
        dt_split = params.get("dataset_split", "none")
        rank = job_config.get("rank", 0)
        rng = params.get("range", -1)

        # --- Base args ---
        if fc == "esft":
            args = (
                f"--fc {fc} "
                f"--thr {fr_or_thr} "
                f"--qb {qb} "
                f"--epochs {params.get('epochs', 1)} "
                f"--dt {dt} "
                f"--dt_split {dt_split} "
                f"--batch_size {params.get('batch_size', 8)} "
                f"--rank {rank} "
            )
        else:
            args = (
                f"--fc {fc} "
                f"--fr {fr_or_thr} "
                f"--qb {qb} "
                f"--lr {params.get('lr', 1e-5)} "
                f"--epochs {params.get('epochs', 1)} "
                f"--dt {dt} "
                f"--dt_split {dt_split} "
                f"--batch_size {params.get('batch_size', 8)} "
                f"--rank {rank} "
            )

        # --- Homogeneous mode: assign start/end for each client ---
        if not hetero_mode and ranges[idx][0] != -1:
            start, end = ranges[idx]
            args += f"--start {start} --end {end} "

        # --- Heterogeneous mode: just pass full split and ignore start/end ---
        if hetero_mode and rng not in (-1, None):
            args += f"--range {rng} "

        args = args.strip()

        runner = ScriptRunner(
            script=train_script,
            script_args=args,
            launch_external_process=True,
            command=f"python3 -u ",
        )

        job.to(runner, site)

        print(f"Added {site} with args: {args}")

    job.export_job("/tmp/nvflare/jobs")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_ids = ",".join(str(i+1) for i in range(n_gpus-1))
        print(f"Running simulator on {n_gpus} available GPUs: {gpu_ids}")
    else:
        print("Warning: No GPUs detected. Running on CPU.")
        gpu_ids = "" # Simulator will run on CPU

    # Pass the list of GPU IDs to the simulator
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu=gpu_ids)


if __name__ == "__main__":
    main()
