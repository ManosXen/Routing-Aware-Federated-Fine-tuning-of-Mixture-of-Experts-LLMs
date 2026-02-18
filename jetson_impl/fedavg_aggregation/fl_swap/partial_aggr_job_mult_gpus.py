from typing import List, Optional

import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
from nvflare.job_config.api import FedJob
from nvflare.apis.job_def import ALL_SITES, SERVER_SITE_NAME
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_int, check_str
from tempfile import TemporaryDirectory
import subprocess
import shlex
import os
import sys
from partial_aggr_controller import PartialAggController
from nvflare.private.fed.app.utils import kill_child_processes
from nvflare.job_config.fed_job_config import FedJobConfig

class PartialAggrJobMultipleGPUs(FedJob):
    def __init__(
        self,
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
        """

        super().__init__(name, min_clients, mandatory_clients)

        self.job: CustomFedJobConfig = CustomFedJobConfig(
            job_name=self.name,
            min_clients=min_clients,
            mandatory_clients=mandatory_clients,
            meta_props=None,
        )

        controller = PartialAggController(
            num_clients=n_clients,
            num_rounds=num_rounds,
            #persistor_id=self.comp_ids["persistor_id"],
        )
        self.to_server(controller)

    def simulator_run(
        self,
        workspace: str,
        n_clients: Optional[int] = None,
        clients: Optional[List[str]] = None,
        threads: Optional[int] = None,
        gpu: Optional[str] = None,
        log_config: Optional[str] = None,
    ):
        """Run the job with the simulator with the `workspace` using `clients` and `threads`.
        For end users.

        Args:
            workspace: workspace directory for job.
            n_clients: number of clients.
            clients: client names.
            threads: number of threads.
            gpu: gpu assignments for simulating clients, comma separated
            log_config: log config mode ('concise', 'default', 'verbose'), filepath, or level

        Returns:
        """
        if clients:
            self.clients = clients
        self._set_all_apps()

        if ALL_SITES in self.clients and not n_clients:
            raise ValueError("Clients were not specified using to(). Please provide the number of clients to simulate.")
        elif ALL_SITES in self.clients and n_clients:
            check_positive_int("n_clients", n_clients)
            self.clients = [f"site-{i}" for i in range(1, n_clients + 1)]
        elif self.clients and n_clients:
            raise ValueError("You already specified clients using `to()`. Don't use `n_clients` in simulator_run.")

        n_clients = len(self.clients)

        if threads is None:
            threads = n_clients


        self.job.simulator_run(
            workspace,
            clients=",".join(self.clients),
            n_clients=n_clients,
            threads=threads,
            gpu=gpu,
            log_config=log_config,
        )

class CustomFedJobConfig(FedJobConfig):
        
    def __init__(self, job_name, min_clients, mandatory_clients=None, meta_props=None) -> None:
        super().__init__(job_name, min_clients, mandatory_clients, meta_props)
  
    def simulator_run(self, workspace, clients=None, n_clients=None, threads=None, gpu=None, log_config=None):
        with TemporaryDirectory() as job_root:
            self.generate_job_config(job_root)

            try:
                command = (
                    f"{sys.executable} fl/custom_simulator/custom_simulator.py "
                    + os.path.join(job_root, self.job_name)
                    + " -w "
                    + workspace
                )
                if clients:
                    clients = self._trim_whitespace(clients)
                    command += " -c " + str(clients)
                if n_clients:
                    command += " -n " + str(n_clients)
                if threads:
                    command += " -t " + str(threads)
                if gpu:
                    command += " -gpu " + str(gpu)
                if log_config:
                    command += " -l" + str(log_config)

                new_env = os.environ.copy()
                process = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=new_env)

                process.wait()

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt, terminate all the child processes.")
                kill_child_processes(os.getpid())
                return -9