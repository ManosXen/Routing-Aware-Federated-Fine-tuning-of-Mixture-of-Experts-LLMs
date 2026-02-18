
from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.app_constant import AppConstants
from typing import List

class FedAvg(BaseFedAvg):
    def run(self) -> None:
        self.info("Start FedAvg.")

        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds

        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            self.info(f"Round {self.current_round} started.")
            model.current_round = self.current_round

            clients = self.sample_clients(self.num_clients)

            results = self.send_model_and_wait(targets=clients, data=model)

            aggregate_results = self.aggregate(results)

            model = self.update_model(model, aggregate_results)

            self.save_model(model)

        self.info("Finished FedAvg.")

    
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
            aggr_result = aggregate_fn(results)
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
