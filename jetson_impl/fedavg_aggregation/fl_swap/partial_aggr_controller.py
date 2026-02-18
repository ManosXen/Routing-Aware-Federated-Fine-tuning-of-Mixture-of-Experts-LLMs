from typing import List
import math
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.abstract.model import make_model_learnable
from partial_aggregator import PartialModelAggregator
from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.security.logging import secure_format_exception
import torch
from nvflare.app_common.workflows.model_controller import ModelController
from pympler import asizeof
from add_frozen_lora_adapter import add_or_update_lora_adapter
import time
import numpy as np 

class PartialAggController(ModelController):
    def __init__(
        self,
        num_clients: int = 3,
        num_rounds: int = 5,
        start_round: int = 0,
        threshold: float=0.1,
        hard_convergence: float=0.6,
        patience: int=1,
        *args,
        **kwargs,
    ):
        """

        A model persistor can be configured via the `persistor_id` argument of the `ModelController`.
        The model persistor is used to load the initial global model which is sent to a list of clients.
        Each client sends it's updated weights after local training which is aggregated.
        Next, the global model is updated.
        The model_persistor will also save the model after training.

        Provides the default implementations for the follow routines:
            - def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel
            - def update_model(self, aggr_result)

        The `run` routine needs to be implemented by the derived class:

            - def run(self)

        Args:
            num_clients (int, optional): The number of clients. Defaults to 3. NOTE: this argument should not be here
            we will remove this argument in next release.
            num_rounds (int, optional): The total number of training rounds. Defaults to 5.
            start_round (int, optional): The starting round number.
        """
        super().__init__(*args, **kwargs)

        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.input_file_name=""
        self.current_round = None

        self.model_params={}
        
        self.prev_params={}
        self.rel_update=[]

        self.patience=patience
        self.threshold=threshold
        self.hard_convergence = hard_convergence
        self.hard_convergenced_experts = []
        self.prev_converged_experts={}
        self.total_weight=0
        self.aggr_weights=[]

        #stats 

        self.client_allocated_gpu=[]
        self.client_reserved_gpu=[]
        self.rel_update_history=[]
        self.lora_experts_count=[]
        self.rec_size=[]
        self.send_size=[]
        self.client_time=[]
        self.server_time=[]
        self.losses=[]
        self.current_lora=[[] for _ in range(16)]
        self.trainable_experts_history=[]

    def total_weight_calc(self, results: List[FLModel]):
        for _result in results:
            client_num = int(_result.meta.get('client_name').split('-')[-1])
            weight=_result.meta.get("num_rows")
            print(f"Client {client_num}: {weight}")
            self.total_weight+=weight
        print(f"Total Weight: {self.total_weight}", flush=True)
            
    @staticmethod
    def _check_results(results: List[FLModel]):
        empty_clients = []
        for _result in results:
            if not _result.params and _result.meta.get("avg_allocated_gpu")!=0 and _result.meta.get("avg_reserved_gpu")!=0:
                #print(f"{_result.trainable}")
                empty_clients.append(_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN))

        if len(empty_clients) > 0:
            raise ValueError(f"Result from client(s) {empty_clients} is empty!")

    @staticmethod
    def aggregate_fn(results: List[FLModel], converged_vals) -> FLModel:
        print(f"Results: {results}")
        if not results:
            raise ValueError("received empty results for aggregation.")
        else:
            print("Everything OK for aggregation")

        aggr_helper = PartialModelAggregator()
        #aggr_metrics_helper = WeightedAggregationHelper()
        #all_metrics = True
        
        rec_size=[]

        for _result in results:
            #print(f"Adding client {_result.meta.get('client_name')} {_result.params, _result.meta.get('trainable')}")
            
            rec = asizeof.asizeof(_result)
            rec_size.append({"client_name" : _result.meta.get('client_name'), "rec" : rec})

            aggr_helper.add(
                data=_result.params,
                trainable=_result.meta.get("trainable"),
                weight=_result.meta.get("num_rows"),
                contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
                contribution_round=_result.current_round,
                client_time=_result.meta.get("time"),
                loss=_result.metrics.get("val_loss"),
                allocated=_result.meta.get("avg_allocated_gpu"),
                reserved=_result.meta.get("avg_reserved_gpu")
            )


            # if not _result.metrics:
            #     all_metrics = False
            # if all_metrics:
            #     aggr_metrics_helper.add(
            #         data=_result.metrics,
            #         weight=_result.meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0),
            #         contributor_name=_result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN),
            #         contribution_round=_result.current_round,
            #     )

        aggr_params, lora_params, current_round_loss, client_times, alloc, res, tr, aggr_weight_dict= aggr_helper.get_result(converged_vals)
        #aggr_metrics = aggr_metrics_helper.get_result() #if all_metrics else None
        
        aggr_result = FLModel(
            params=aggr_params,
            params_type=results[0].params_type,
            #metrics=aggr_metrics,
            meta={"nr_aggregated": len(results), "current_round": results[0].current_round, "lora_params" : lora_params},
        )
        
        return aggr_result, current_round_loss, rec_size, client_times, alloc, res, tr, aggr_weight_dict
        


    def aggregate(self, results: List[FLModel], aggregate_fn=None) -> FLModel:
        """Called by the `run` routine to aggregate the training results of clients.

        Args:
            results: a list of FLModel containing training results of the clients.
            aggregate_fn: a function that turns the list of FLModel into one resulting (aggregated) FLModel.

        Returns: aggregated FLModel.

        """
        self.debug("Start aggregation.")
        self.event(AppEventType.BEFORE_AGGREGATION)
        self._check_results(results)

        if not aggregate_fn:
            aggregate_fn = self.aggregate_fn

        self.info(f"aggregating {len(results)} update(s) at round {self.current_round}")
        try:
            aggr_result, current_round_loss, rec, times, alloc, res, tr, self.aggr_weights= aggregate_fn(results, self.prev_converged_experts)
            self.losses.append(current_round_loss)
            self.rec_size.append(rec)
            self.client_time.append(times)
            self.client_allocated_gpu.append(alloc)
            self.client_reserved_gpu.append(res)
            self.trainable_experts_history.append(tr)
        except Exception as e:
            error_msg = f"Exception in aggregate call: {secure_format_exception(e)}"
            self.exception(error_msg)
            self.panic(error_msg)
            return FLModel()
        self._results = []

        self.fire_event_with_data(
            AppEventType.AFTER_AGGREGATION, self.fl_ctx, AppConstants.AGGREGATION_RESULT, aggr_result
        )

        self.debug("End aggregation.")
        return aggr_result

    def update_model(self, aggr_result):
        """Called by the `run` routine to update the current global model (self.model) given the aggregated result.

        Args:
            model: FLModel to be updated.
            aggr_result: aggregated FLModel.

        Returns: None.

        """
        self.event(AppEventType.BEFORE_SHAREABLE_TO_LEARNABLE)

        lora_params=aggr_result.meta.get("lora_params")
        print(f"LoRA experts: {sum(len(x) for x in lora_params)}")

        #print(f"FL aggregated {aggr_result.params} {aggr_result.meta.get('lora_params')}")

        
        for name, val in aggr_result.params.items():
            self.model_params[name]=val

        self.lora_experts_count.append(len(self.model_params.keys()))
        
        for i in range(16):
            self.current_lora[i] = list(set(lora_params[i]) | set(self.current_lora[i]))

        #model = FLModelUtils.update_model(model, aggr_result)
        print('end update')
        # persistor uses Learnable format to save model
        #ml = make_model_learnable(weights=self.model_params)
        #self.fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, ml, private=True, sticky=True)

        self.event(AppEventType.AFTER_SHAREABLE_TO_LEARNABLE)

    def norm_2(self, x):
        return torch.linalg.norm(x, ord=2).item()
    
    def check_convergence(self, last_rel_update, rec_params):

        converged_experts=[]

        print(self.rel_update, flush=True)

        rec_params_transformed = {} #keys expert global idx, val dictionary of this expert's params name : val 
        
        for name, val in rec_params.items():
            name_aux=name.split('.')
            layer_idx = int(name_aux[4])
            expert_idx = int(name_aux[7])
            global_idx = layer_idx*64+expert_idx
            if global_idx not in rec_params_transformed.keys():
                rec_params_transformed[global_idx] ={}
            rec_params_transformed[global_idx][name]=val

        for expert in last_rel_update.keys():
            counter=0
            for r in range(0, self.patience+1):
                if expert in self.rel_update[-1-r].keys():
                    if self.rel_update[-1-r][expert]<=self.threshold:
                        counter+=1
                    else:
                        break
                    
            if counter>=self.patience+1:
                converged_experts.append(expert)
                for name, val in rec_params_transformed[expert].items():
                    self.prev_converged_experts[name]={"value" : val, "aggr_weight": self.aggr_weights[name]}    
                    if self.aggr_weights[name]/self.total_weight > self.hard_convergence:
                        print(f"Hard Converged Expert {expert} weight: {self.aggr_weights[name]/self.total_weight}", flush=True)
                        self.hard_convergenced_experts.append(expert)
                        break

        del self.rel_update[0]

        return converged_experts

    def rel_update_calc_and_check_convergence(self, aggr_result):

        ###DEBUG
        #print(f"Rec params: \n{aggr_result.params}")

        if self.current_round==0:
            self.prev_params=aggr_result.params
            return []

        epsilon=1e-8
        
        rec_params=aggr_result.params
        
        param_rel_updates={}
        for name in rec_params.keys():

            if name in self.prev_params.keys():
                delta_theta_cons = rec_params[name] - self.prev_params[name]
                
                norm_delta = self.norm_2(delta_theta_cons)
                norm_theta = self.norm_2(rec_params[name])
                
                rel_update = norm_delta / (norm_theta + epsilon)
                param_rel_updates[name]=rel_update

        
        expert_rel_updates={}
        for name, update in param_rel_updates.items():
            name_aux=name.split('.')
            layer_idx = int(name_aux[4])
            expert_idx = int(name_aux[7])
            global_idx=layer_idx*64+expert_idx
            
            if global_idx not in expert_rel_updates.keys():
                expert_rel_updates[global_idx] = 0.0
            
            expert_rel_updates[global_idx] += update**2

        

        for expert in expert_rel_updates.keys():
            expert_rel_updates[expert] = math.sqrt(expert_rel_updates[expert] / 6.0)

        self.rel_update.append(expert_rel_updates)
        self.rel_update_history.append(expert_rel_updates)

        self.prev_params=rec_params
        
        if self.current_round>=self.start_round+self.patience+1:
            return self.check_convergence(expert_rel_updates, rec_params)
        else:
            return []
    
    def calculate_fr(self):
        clients = self.sample_clients(self.num_clients)

        # fixed freeze rate for all
        if self.min_fr == self.max_fr:
            for client in clients:
                self.client_fr[client] = self.min_fr
            return
        
        starting_fr = (self.min_fr + self.max_fr) / 2

        req_prof = FLModel(params={}, current_round=-1, meta={"fr": starting_fr})
        results = self.send_model_and_wait(targets=clients, data=req_prof)

        print(f"Aggregating {len(results)} for freeze rate profiling...", flush=True)

        fr_prof_time = {}
        for _result in results:
            cl_time = _result.meta.get("fr_prof_time")
            cl_name = _result.meta.get("client_name", AppConstants.CLIENT_UNKNOWN)
            fr_prof_time[cl_name] = cl_time
        print(f"Profiling times: {fr_prof_time}", flush=True)

        times = np.array(list(fr_prof_time.values()), dtype=float)

        # Compute Z-scores
        mu = np.mean(times)
        sigma = np.std(times)
        if sigma == 0:
            sigma = 1e-8  # avoid division by zero
        z_scores = (times - mu) / sigma

        # Clip Z-scores to [-2, 2]
        z_scores_clipped = np.clip(z_scores, -2.0, 2.0)

        # Normalize clipped Z-scores to [0, 1]
        z_min, z_max = -2.0, 2.0
        norm = (z_scores_clipped - z_min) / (z_max - z_min)

        fr_values = self.min_fr + norm * (self.max_fr - self.min_fr)

        for client, fr in zip(fr_prof_time.keys(), fr_values):
            self.client_fr[client] = float(fr)

        print(f"Assigned freeze rates: {self.client_fr}", flush=True)


    
    def run(self) -> None:

        #model = self.load_model()
        start_round = self.start_round
        total_rounds = self.num_rounds
        converged_experts=[]

        #self.calculate_fr()

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            print(f"Round {self.current_round} started.", flush=True)

            clients = self.sample_clients(self.num_clients)

            print(f"Memory {self.client_allocated_gpu}, {self.client_reserved_gpu}", flush=True)
            print(f"Trainable Expert History: {self.trainable_experts_history}", flush=True)
            
            if self.current_round==self.start_round:
                send_result = FLModel(params={}, current_round=self.current_round)
            else:
                print('else')

                numpy_params = {
                    k: v.detach().cpu().numpy() 
                    for k, v in aggregate_results.params.items()
                }
                
                numpy_lora_params = None
                lora_params = aggregate_results.meta.get("lora_params")
                print(f"lora_params: {lora_params}", flush=True)

                numpy_lora_params = lora_params 

                send_result = FLModel(
                    params=numpy_params,
                    current_round=self.current_round,
                    meta={"lora_params": numpy_lora_params, "converged_experts" : converged_experts, "hard_converged_experts" : self.hard_convergenced_experts}
                )

            sent_size = asizeof.asizeof(send_result)
            self.send_size.append(sent_size)

            #print('Sending...')

            end_time=time.time()

            if self.current_round!=self.start_round:
                self.server_time.append(end_time-start_time)

            results = self.send_model_and_wait(targets=clients, data=send_result)
            
            start_time=time.time()
            
            if self.current_round==0:
                save_file=results[0].meta.get("save_file_ext")
                self.total_weight_calc(results)
            aggregate_results = self.aggregate(
                results, aggregate_fn=self.aggregate_fn
            )

            converged_experts=self.rel_update_calc_and_check_convergence(aggregate_results)
            print(f"Converged experts in round {self.current_round}: {converged_experts}", flush=True)

            self.update_model(aggregate_results)

            save_path = f"/files/jetson_impl/fedavg_aggregation/fl_swap/stats/{save_file}_params_{self.current_round}.pt" 
            print(f"Params save file: {save_path}", flush=True)
            torch.save(self.model_params, save_path)

            end_time=time.time()

        stats_save_path = f"/files/jetson_impl/fedavg_aggregation/fl_swap/stats/training_stats_{save_file}.txt"
        with open(stats_save_path, "w") as f:
                f.write("Losses:\n")
                f.write(f"{str(self.losses)}\n\n")
                f.write("Current LoRA:\n")
                f.write(f"{str(self.current_lora)}\n\n")
                f.write("LoRA Expert History:\n")
                f.write(f"{str(self.lora_experts_count)}\n\n")                
                f.write("Size of data sent from the server:\n")
                f.write(f"{str(self.send_size)}\n\n")
                f.write("Size of data sent from the clients:\n")
                f.write(f"{str(self.rec_size)}\n\n")
                f.write("Client training time:\n")
                f.write(f"{str(self.client_time)}\n\n")
                f.write("Server Aggregation time:\n")
                f.write(f"{str(self.server_time)}\n\n")
                f.write("Allocated GPU VRAM:\n")
                f.write(f"{str(self.client_allocated_gpu)}\n\n")
                f.write("Reserved GPU VRAM:\n")
                f.write(f"{str(self.client_reserved_gpu)}\n\n")
                f.write("Relative Update History:\n")
                f.write(f"{str(self.rel_update_history)}\n\n")
                f.write("Trainable Experts History:\n")
                f.write(f"{str(self.trainable_experts_history)}\n\n")