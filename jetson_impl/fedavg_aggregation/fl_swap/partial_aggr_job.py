from typing import List, Optional

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from partial_aggr_controller import PartialAggController

class PartialAggrJob(FedJob):
    def __init__(
        self,
        n_clients: int,
        num_rounds: int,
        threshold: float=0.1,
        patience : int=1,
        hard_convergence : float=0.7,
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
        """

        super().__init__(name, min_clients, mandatory_clients)
        controller = PartialAggController(
            num_clients=n_clients,
            num_rounds=num_rounds,
            threshold=threshold,
            patience=patience,
            hard_convergence=hard_convergence
        )
        self.to_server(controller)