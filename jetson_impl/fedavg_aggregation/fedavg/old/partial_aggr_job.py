from typing import List, Optional

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from partial_aggr_controller import PartialAggController

class PartialAggrJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module,
        n_clients: int,
        num_rounds: int,
        name: str = "fed_job",
        min_clients: int = 1,
        mandatory_clients: Optional[List[str]] = None,
        key_metric: str = "accuracy",
    ):
        """PyTorch FedAvg Job.

        Configures server side FedAvg controller, persistor with initial model, and widgets.

        User must add executors.

        Args:
            initial_model (nn.Module): initial PyTorch Model
            n_clients (int): number of clients for this job
            num_rounds (int): number of rounds for FedAvg
            name (name, optional): name of the job. Defaults to "fed_job"
            min_clients (int, optional): the minimum number of clients for the job. Defaults to 1.
            mandatory_clients (List[str], optional): mandatory clients to run the job. Default None.
            key_metric (str, optional): Metric used to determine if the model is globally best.
                if metrics are a `dict`, `key_metric` can select the metric used for global model selection.
                Defaults to "accuracy".
        """
        if not isinstance(initial_model, nn.Module):
            raise ValueError(f"Expected initial model to be nn.Module, but got type f{type(initial_model)}.")

        super().__init__(initial_model, name, min_clients, mandatory_clients, key_metric)

        controller = PartialAggController(
            num_clients=n_clients,
            num_rounds=num_rounds,
            persistor_id=self.comp_ids["persistor_id"],
        )
        self.to_server(controller)