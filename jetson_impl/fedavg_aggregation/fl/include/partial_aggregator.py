import numpy as np
import re
import threading
from typing import Optional
import torch

class PartialModelAggregator(object):

    def __init__(self):
        """
        Initialize the aggregator.
        We need to store the client contributions and the global model state.
        """
        super().__init__()
        self.lock = threading.Lock()
        self.params_dict = {}
        self.lora_params=[]
        self.weights = {}

    def add(self, data, trainable, weight, contributor_name, contribution_round):

        print(f"Adding weights of client {contributor_name} for round {contribution_round}")
        
        for name, val in data.items():
            with self.lock:
                if name not in self.params_dict:
                    self.params_dict[name] = []
                self.params_dict[name].append((val, weight))
        
        for layer_idx, tr_l in enumerate(trainable):
            if layer_idx >= len(self.lora_params):
                self.lora_params.append([])
            for tr_exp in tr_l:
                self.lora_params[layer_idx].append(tr_exp)


    def get_result(self):
            
            aggregated_dict = {}

            for name, lst in self.params_dict.items():
                weight_total = sum(w for _, w in lst)
                
                val0_np = lst[0][0]
                agg_val = torch.zeros_like(torch.from_numpy(val0_np))

                for val_np, w in lst:
                    agg_val += (w / weight_total) * torch.from_numpy(val_np)

                aggregated_dict[name] = agg_val

            return aggregated_dict, self.lora_params