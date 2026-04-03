"""Flower server strategy for the federated face recognition system."""

from __future__ import annotations

from typing import Callable

import flwr as fl
from flwr.common import Metrics, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from federated_project.federation import (
    ClientUpdate,
    aggregate_client_updates,
    create_model,
    get_global_parameters,
    split_client_update_parameters,
)


class SpreadoutFedAvg(FedAvg):
    """
    Flower FedAvg strategy customized for the README pipeline.

    Clients receive the full global model, train locally, and return:
    1. updated FaceNet feature extractor weights
    2. their own class embedding row only

    The server then averages the feature extractor, restores each embedding row
    into the global matrix, and optionally applies spreadout regularization.
    """

    def __init__(
        self,
        num_clients: int,
        pretrained: str = "vggface2",
        spreadout_strength: float = 0.0,
        spreadout_margin: float = 0.35,
        spreadout_steps: int = 1,
        spreadout_lr: float = 0.1,
        **kwargs: object,
    ) -> None:
        self.model = create_model(num_clients=num_clients, pretrained=pretrained, device="cpu")
        self.spreadout_strength = spreadout_strength
        self.spreadout_margin = spreadout_margin
        self.spreadout_steps = spreadout_steps
        self.spreadout_lr = spreadout_lr

        kwargs.setdefault(
            "initial_parameters",
            ndarrays_to_parameters(get_global_parameters(self.model)),
        )
        super().__init__(**kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.FitRes]],
        failures: list[tuple[ClientProxy, fl.common.FitRes] | BaseException],
    ) -> tuple[fl.common.Parameters | None, dict[str, Scalar]]:
        """Aggregate Flower fit results using the project-specific protocol."""
        if not results:
            return None, {}
        if failures and not self.accept_failures:
            return None, {}

        client_updates: list[ClientUpdate] = []
        for _client_proxy, fit_res in results:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            backbone_parameters, class_embedding = split_client_update_parameters(
                self.model,
                parameters,
            )

            client_updates.append(
                ClientUpdate(
                    client_id=int(fit_res.metrics["client_id"]),
                    num_examples=fit_res.num_examples,
                    feature_extractor_parameters=backbone_parameters,
                    class_embedding=class_embedding,
                    loss=float(fit_res.metrics["loss"]) if "loss" in fit_res.metrics else None,
                )
            )

        metrics = aggregate_client_updates(
            self.model,
            client_updates,
            spreadout_margin=self.spreadout_margin,
            spreadout_strength=self.spreadout_strength,
            spreadout_steps=self.spreadout_steps,
            spreadout_lr=self.spreadout_lr,
        )
        metrics["server_round"] = float(server_round)

        global_parameters = ndarrays_to_parameters(get_global_parameters(self.model))
        return global_parameters, metrics


def build_fit_config_fn(
    local_epochs: int,
    learning_rate: float,
    margin: float,
) -> Callable[[int], dict[str, Scalar]]:
    """Create the configuration callback sent by Flower before each round."""

    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": learning_rate,
            "margin": margin,
        }

    return fit_config


def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Average scalar client metrics so Flower can report round summaries."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    losses = [
        float(client_metrics["loss"]) * num_examples
        for num_examples, client_metrics in metrics
        if "loss" in client_metrics
    ]
    return {
        "loss": float(sum(losses) / total_examples) if losses else 0.0,
    }


def create_server_strategy(
    num_clients: int,
    pretrained: str = "vggface2",
    fraction_fit: float = 1.0,
    min_fit_clients: int | None = None,
    min_available_clients: int | None = None,
    local_epochs: int = 1,
    learning_rate: float = 1e-3,
    margin: float = 0.5,
    spreadout_strength: float = 0.0,
    spreadout_margin: float = 0.35,
    spreadout_steps: int = 1,
    spreadout_lr: float = 0.1,
    accept_failures: bool = False,
) -> SpreadoutFedAvg:
    """Create the Flower strategy with sensible defaults for this project."""
    return SpreadoutFedAvg(
        num_clients=num_clients,
        pretrained=pretrained,
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients if min_fit_clients is not None else num_clients,
        min_available_clients=(
            min_available_clients if min_available_clients is not None else num_clients
        ),
        on_fit_config_fn=build_fit_config_fn(
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            margin=margin,
        ),
        fit_metrics_aggregation_fn=weighted_average_metrics,
        spreadout_strength=spreadout_strength,
        spreadout_margin=spreadout_margin,
        spreadout_steps=spreadout_steps,
        spreadout_lr=spreadout_lr,
        accept_failures=accept_failures,
    )


def start_flower_server(
    server_address: str,
    num_rounds: int,
    strategy: SpreadoutFedAvg,
) -> None:
    """Start the Flower aggregation server."""
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
