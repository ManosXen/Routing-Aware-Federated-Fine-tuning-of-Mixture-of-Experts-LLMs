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
        self.losses={}
        self.time={}
        self.alloc={}
        self.res={}
        self.trainable_experts={}

    def add(self, data, trainable, weight, contributor_name, contribution_round, client_time, loss, allocated, reserved):

        print(f"Adding weights of client {contributor_name} for round {contribution_round}, weight {weight}\ntrainable {trainable}")
        
        self.losses[contributor_name]=loss
        for name, val in data.items():
            with self.lock:
                if name not in self.params_dict:
                    self.params_dict[name] = []
                self.params_dict[name].append((val, weight))
        
        for layer_idx, tr_l in enumerate(trainable):
            if layer_idx >= len(self.lora_params):
                self.lora_params.append([])
            for tr_exp in tr_l:
                if tr_exp not in self.lora_params[layer_idx]:
                    self.lora_params[layer_idx].append(tr_exp)

        self.time[contributor_name]=client_time
        self.alloc[contributor_name]=allocated
        self.res[contributor_name]=reserved
        self.trainable_experts[contributor_name]=sum(len(x) for x in trainable)

    def get_result(self, converged_vals):
            
            print("Calculating result...")
            # print(self.lora_params)
            # for name, val in self.params_dict.items():
            #     print(f"{name}: {len(val)}")
            aggregated_dict = {}
            aggr_weight_dict = {}

            for name, lst in self.params_dict.items():
                weight_total = sum(w for _, w in lst)

                if name in converged_vals.keys():
                    weight_total+=converged_vals[name]["aggr_weight"]
                
                val0_np = lst[0][0]
                agg_val = torch.zeros_like(torch.from_numpy(val0_np))
                
                #σprint(name)
                for val_np, w in lst:
                    #print(f"Weight coefficient: {w/weight_total}")
                    agg_val += (w / weight_total) * torch.from_numpy(val_np)

                if name in converged_vals.keys():
                    w = converged_vals[name]["aggr_weight"]
                    agg_val += (w / weight_total) * converged_vals[name]["value"] #εδώ υπάρχει περίπτωση να μην χρειάζεται το torch.from_numpy
                

                name_aux=name.split('.')
                layer_idx = int(name_aux[4])
                expert_idx = int(name_aux[7])

                aggregated_dict[name] = agg_val
                aggr_weight_dict[name] = weight_total

            return aggregated_dict, self.lora_params, self.losses, self.time, self.alloc, self.res, self.trainable_experts, aggr_weight_dict