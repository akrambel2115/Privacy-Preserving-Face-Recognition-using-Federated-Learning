"""Flower server strategy for the federated face recognition system."""

from __future__ import annotations

from typing import Callable

import flwr as fl
from flwr.common import (
    Context,
    FitIns,
    Metrics,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from federated_project.federation import (
    ClientUpdate,
    aggregate_client_updates,
    create_model,
    get_feature_extractor_parameters,
    get_global_parameters,
    set_feature_extractor_parameters,
    split_client_update_parameters,
)


class SpreadoutFedAvg(FedAvg):
    """
    Flower FedAvg strategy customized for the README pipeline.

    In legacy mode, clients receive the full global model, train locally, and return:
    1. updated FaceNet feature extractor weights
    2. their own class embedding row only

    The server then averages the feature extractor, restores each embedding row
    into the global matrix, and optionally applies spreadout regularization.
    In secure mode, clients exchange only the shared FaceNet backbone; personal
    embedding rows stay on client devices.
    """

    def __init__(
        self,
        num_clients: int,
        pretrained: str = "vggface2",
        spreadout_strength: float = 0.0,
        spreadout_margin: float = 0.35,
        spreadout_steps: int = 1,
        spreadout_lr: float = 0.1,
        secure_aggregation: bool = False,
        train_backbone: bool = False,
        **kwargs: object,
    ) -> None:
        self.model = create_model(
            num_clients=num_clients,
            pretrained=pretrained,
            device="cpu",
            train_backbone=train_backbone,
        )
        self.spreadout_strength = spreadout_strength
        self.spreadout_margin = spreadout_margin
        self.spreadout_steps = spreadout_steps
        self.spreadout_lr = spreadout_lr
        self.secure_aggregation = secure_aggregation

        initial_ndarrays = (
            get_feature_extractor_parameters(self.model)
            if self.secure_aggregation
            else get_global_parameters(self.model)
        )
        kwargs.setdefault("initial_parameters", ndarrays_to_parameters(initial_ndarrays))
        super().__init__(**kwargs)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Add secure aggregation metadata after Flower samples clients."""
        configured_clients = super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )
        if not self.secure_aggregation:
            return configured_clients

        for _client_proxy, fit_ins in configured_clients:
            fit_ins.config["secure_aggregation"] = True
        return configured_clients

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

        if self.secure_aggregation:
            aggregated_parameters = parameters_to_ndarrays(results[0][1].parameters)
            set_feature_extractor_parameters(self.model, aggregated_parameters)
            metrics = _aggregate_secure_fit_metrics(results)
            metrics["server_round"] = float(server_round)
            return ndarrays_to_parameters(get_feature_extractor_parameters(self.model)), metrics

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
    preservation_strength: float,
    negative_strength: float,
    negative_margin: float,
) -> Callable[[int], dict[str, Scalar]]:
    """Create the configuration callback sent by Flower before each round."""

    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
            "lr": learning_rate,
            "margin": margin,
            "preservation_strength": preservation_strength,
            "negative_strength": negative_strength,
            "negative_margin": negative_margin,
        }

    return fit_config


def _aggregate_secure_fit_metrics(
    results: list[tuple[ClientProxy, fl.common.FitRes]],
) -> dict[str, Scalar]:
    return {
        "secure_aggregation": True,
    }


def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Average scalar client metrics so Flower can report round summaries."""
    total_examples = sum(
        int(client_metrics.get("num_samples", num_examples))
        for num_examples, client_metrics in metrics
    )
    if total_examples == 0:
        return {}

    losses = [
        float(client_metrics["loss"])
        * int(client_metrics.get("num_samples", num_examples))
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
    secure_aggregation: bool = False,
    train_backbone: bool = False,
    preservation_strength: float = 0.0,
    negative_strength: float = 0.0,
    negative_margin: float = 0.2,
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
            preservation_strength=preservation_strength,
            negative_strength=negative_strength,
            negative_margin=negative_margin,
        ),
        fit_metrics_aggregation_fn=weighted_average_metrics,
        spreadout_strength=spreadout_strength,
        spreadout_margin=spreadout_margin,
        spreadout_steps=spreadout_steps,
        spreadout_lr=spreadout_lr,
        accept_failures=accept_failures,
        secure_aggregation=secure_aggregation,
        train_backbone=train_backbone,
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


def _run_config_value(context: Context, key: str, default: object) -> object:
    return context.run_config[key] if key in context.run_config else default


def _run_config_bool(context: Context, key: str, default: bool) -> bool:
    value = _run_config_value(context, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Flower ServerApp entrypoint using the built-in SecAgg+ workflow."""
    num_clients = int(_run_config_value(context, "num-clients", 2))
    num_rounds = int(_run_config_value(context, "num-server-rounds", 3))

    strategy = create_server_strategy(
        num_clients=num_clients,
        pretrained=str(_run_config_value(context, "pretrained", "vggface2")),
        fraction_fit=float(_run_config_value(context, "fraction-fit", 1.0)),
        min_fit_clients=int(_run_config_value(context, "min-fit-clients", num_clients)),
        min_available_clients=int(
            _run_config_value(context, "min-available-clients", num_clients)
        ),
        local_epochs=int(_run_config_value(context, "local-epochs", 1)),
        learning_rate=float(_run_config_value(context, "learning-rate", 1e-3)),
        margin=float(_run_config_value(context, "margin", 0.5)),
        spreadout_strength=float(_run_config_value(context, "spreadout-strength", 0.0)),
        spreadout_margin=float(_run_config_value(context, "spreadout-margin", 0.35)),
        spreadout_steps=int(_run_config_value(context, "spreadout-steps", 1)),
        spreadout_lr=float(_run_config_value(context, "spreadout-lr", 0.1)),
        accept_failures=_run_config_bool(context, "accept-failures", False),
        secure_aggregation=True,
        train_backbone=_run_config_bool(context, "train-backbone", False),
        preservation_strength=float(_run_config_value(context, "preservation-strength", 0.0)),
        negative_strength=float(_run_config_value(context, "negative-strength", 0.0)),
        negative_margin=float(_run_config_value(context, "negative-margin", 0.2)),
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    clipping_range = max(
        float(_run_config_value(context, "secure-clipping-range", 64.0)),
        float(2 * num_clients + 1),
    )
    fit_workflow = SecAggPlusWorkflow(
        num_shares=_run_config_value(context, "num-shares", 3),
        reconstruction_threshold=_run_config_value(
            context,
            "reconstruction-threshold",
            2,
        ),
        max_weight=float(_run_config_value(context, "secure-max-weight", 1.0)),
        clipping_range=clipping_range,
        quantization_range=int(_run_config_value(context, "quantization-range", 4194304)),
        modulus_range=int(_run_config_value(context, "modulus-range", 4294967296)),
        timeout=float(_run_config_value(context, "secure-timeout", 60.0)),
    )
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)
