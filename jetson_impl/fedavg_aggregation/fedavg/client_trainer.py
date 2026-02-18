import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset, load_from_disk
import argparse
import time
import gc
import os
import numpy as np 
import pickle
import json

# NVFlare imports
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.apis.fl_context import FLContext

# Import your custom functions (Keep your existing imports)
from build_model import build_model
from train_v2 import train
from activation_freeze import freeze_experts_activations
from gradient_freeze import freeze_experts_gradients
from esft import freeze_experts_esft
from prepare_reprofiling import prepare_reprofiling
from python_filter import is_python_example
from prompt_formation import * 
from no_trainable_params_exception import NoTrainableParams

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PartialClientTrainer(Executor):
    def __init__(
        self,
        fc: str,
        fr_or_thr: float,
        qb: int,
        lr: float,
        epochs: int,
        dt: str,
        batch_size: int,
        rank: int,
        save_file_aux: str,
        dataset_file: str,
    ):
        super().__init__()
        
        # --- Store all arguments ---
        self.fc = fc
        self.fr_or_thr = fr_or_thr
        self.qb = qb
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.rank = rank
        root, _ = os.path.splitext(save_file_aux)
        self.save_file_aux = root
        self.dataset_file = dataset_file
        self.dt=dt
        
        # --- Client-side state ---
        self.client_lora_weights = {}
        self.current_lora = []
        self.trainable = []
        self.model = None
        self.tokenizer = None

        self.effective_batch_size=32

        print("="*20)
        print(f"Init Called! Save file: {self.save_file_aux}")
        print("="*20)
        
    def dataset_load(self, client_name):

        print(f"Loading dataset: {self.dataset_file}")
    
        with open(self.dataset_file, 'r') as f:
            d=json.load(f)
            selected_indices=d["clients"][int(client_name[-1])]

        dataset=load_dataset(self.dt)
        dataset = dataset["train"] if "train" in dataset else dataset

        dataset=dataset.select(selected_indices)

        self.train_dataset = dataset.shuffle()
        #self.val_dataset = dataset["test"].shuffle()
        print("Dataset loaded and split.")
        
        # --- Select Prompt Function ---
        if "commonsense_qa" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_commonsense_qa
        elif "boolq" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_boolq
        elif "nli" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_nli
        elif "winogrande" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_winogrande
        elif "social_i_qa" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_social_iqa
        elif "piqa" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_piqa
        elif "race" in self.dt:
            self.prompt_fn = prompt_formation_and_tokenize_race
        else:
            raise ValueError("Error in prompt formation")
        
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        
        # 1. --- Unpack model from server ---
        # This is the correct way to receive data from a ModelController
        try:
            fl_input = FLModelUtils.from_shareable(shareable)
            current_round = fl_input.current_round
            client_name = fl_ctx.get_identity_name()
            print(f"\n===== Client {client_name} starting round {current_round} =====", flush=True)
            state_file = f"/files/jetson_impl/fedavg_aggregation/fedavg/temp/{self.save_file_aux}_{fl_ctx.get_identity_name()}_state.pkl"
            
            if current_round > 0:
                if os.path.exists(state_file):
                    print(f"Loading preserved state from {state_file}...", flush=True)
                    with open(state_file, 'rb') as f:
                        saved_state = pickle.load(f)
                        
                    self.client_lora_weights = saved_state.get('client_lora_weights', {})
                    self.current_lora = saved_state.get('current_lora', [])
                    self.trainable = saved_state.get('trainable', [])
                    print("State successfully restored.", flush=True)
                else:
                    print(f"[WARNING] Round > 0 but no state file found at {state_file}", flush=True)
            
        except Exception as e:
            print(f"Error unpacking shareable: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
        self.dataset_load(client_name)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        try:
            # 2. --- Run Client Logic ---
            if current_round == 0:
                print(f"\n===== Client {client_name} starting round {current_round} =====", flush=True)
                print("Round 0: Building model and profiling...")
                self.model, self.tokenizer = build_model(
                    "allenai/OLMoE-1B-7B-0125", self.qb, self.rank, load_prev_weights=False
                )

                self.trainable=[[i for i in range(64)] for _ in range(16)]
                self.current_lora = [list(x) for x in self.trainable]
                agg_time = 0.0

                print(self.trainable)

            else:
                print(f"\n===== Client {client_name} starting round {current_round} =====", flush=True)
                agg_start = time.time()
                
                # --- AGGREGATION ---
                received_params = fl_input.params
                received_params = {k: torch.tensor(v) for k, v in received_params.items()} # Convert numpy back to tensor
                lora_params = fl_input.meta.get("lora_params")
                
                if lora_params is None:
                    print("[ERROR] Server did not send 'lora_params' in meta.", flush=True)
                else:
                    for name, val in received_params.items():
                        self.client_lora_weights[name] = val
                    
                    if len(lora_params) != 16:
                        print(f"[WARNING] lora_params len={len(lora_params)} (expected 16)", flush=True)

                    for i in range(16):
                        if i >= len(self.current_lora) or i >= len(lora_params):
                            break
                        self.current_lora[i] = list(set(lora_params[i]) | set(self.current_lora[i]))

                agg_end = time.time()
                agg_time = agg_end - agg_start

                tr = sum(len(x) for x in self.trainable)
                print(f"Client {client_name} trainable params in round {current_round-1}: {tr}")

                tr = sum(len(x) for x in self.trainable)
                print(f"Client {client_name} trainable params in round {current_round}: {tr}")
                self.model, self.tokenizer = build_model(
                    "allenai/OLMoE-1B-7B-0125",
                    self.qb,
                    self.rank,
                    load_prev_weights=True,
                    client_lora_params=self.current_lora,
                    current_lora_weights=self.client_lora_weights,
                )

                del received_params, lora_params

                # if self.fc == "grad_freeze" and current_round % 2 == 0:
                #     print(f"Reprofiling for round={current_round}\n")
                #     prepare_reprofiling(self.model, self.rank)
                #     self.trainable, sorted_indx = freeze_experts_gradients(self.model, self.fr_or_thr, self.train_dataset, self.prompt_fn, self.rank, current_round)
                #     for p in self.model.parameters():
                #         p.grad = None

            self.model.to(DEVICE)
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            # --- TRAINING ---
            print("Starting local training...", flush=True)
            train_start = time.time()
            val_loss, mem_stats = train(
                self.model, self.train_dataset, self.dt,
                self.prompt_fn, current_round, self.fc, self.epochs,
                self.lr, self.fr_or_thr, self.fc, self.qb, self.rank, self.batch_size,
            )
            torch.cuda.synchronize()
            train_end = time.time()
            train_time = train_end - train_start                

            stats_array = np.array(mem_stats)
    
            avg_allocated = np.mean(stats_array[:, 2]).item()
            
            avg_reserved = np.mean(stats_array[:, 3]).item()

            # 3. --- Create Output FLModel ---
            # Get parameters as Tensors. to_shareable() will handle conversion.
            output_params = {name: p.cpu().detach().numpy() for name, p in self.model.named_parameters() if p.requires_grad}

            output_model = FLModel(
                params=output_params,
                current_round=current_round,
                metrics={"val_loss": val_loss},
                meta={
                    "client_name": client_name,
                    "num_rows": len(self.train_dataset),
                    "trainable": self.trainable,
                    "rank": self.rank,
                    "time": (train_time + agg_time),
                    "avg_allocated_gpu" : avg_allocated,
                    "avg_reserved_gpu" : avg_reserved,
                },
            )
            
            if current_round == 0:
                output_model.meta["save_file_ext"] = self.save_file_aux

            # 4. --- Pack and Return Shareable ---
            # This is the correct way to send data to a ModelController
            print("Packing results into Shareable...", flush=True)
            return FLModelUtils.to_shareable(output_model)
        
        except NoTrainableParams as e:
            print(f"No trainable parameters in round {current_round} for client {client_name}", flush=True)
            output_model = FLModel(
                params={},
                current_round=current_round,
                metrics={"val_loss": 0.0},
                meta={
                    "client_name": client_name,
                    "num_rows": len(self.train_dataset),
                    "trainable": self.trainable,
                    "rank": self.rank,
                    "time": (agg_time),
                    "avg_allocated_gpu" : 0.0,
                    "avg_reserved_gpu" : 0.0,
                },
            )

            print("Packing results into Shareable...", flush=True)
            return FLModelUtils.to_shareable(output_model)

        except Exception as e:
            print(f"Exception during client execution: {e}", flush=True)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        
        finally:
            # 5. --- Cleanup ---
            print("Releasing GPU memory...", flush=True)
            del self.model, self.tokenizer
            self.model = None
            self.tokenizer = None
            dataset = None
            self.train_dataset = None
            self.val_dataset = None

            print(f"Performing state backup for job {fl_ctx.get_job_id()} client {fl_ctx.get_identity_name()}...", flush=True)
            
            # IMPORTANT: Move tensors to CPU before saving to pickle. 
            # This prevents CUDA errors when reloading on a different GPU process.
            cpu_client_lora_weights = {}
            for k, v in self.client_lora_weights.items():
                if isinstance(v, torch.Tensor):
                    cpu_client_lora_weights[k] = v.cpu()
                else:
                    cpu_client_lora_weights[k] = v

            state_data = {
                'client_lora_weights': cpu_client_lora_weights,
                'current_lora': self.current_lora,
                'trainable': self.trainable,
            }

            state_file = f"/files/jetson_impl/fedavg_aggregation/fedavg/temp/{self.save_file_aux}_{fl_ctx.get_identity_name()}_state.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state_data, f)
            print(f"State saved to {state_file}", flush=True)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU memory released.", flush=True)