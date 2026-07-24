"""Versioned configuration contracts for private Intuitive experiments."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from pipeline.clustering import (
    DEFAULT_INTUITIVE_EMPTY_CLUSTER_POLICY,
    DEFAULT_INTUITIVE_INIT_STRATEGY,
    DEFAULT_INTUITIVE_MIN_CLUSTER_SIZE,
    DEFAULT_INTUITIVE_STRICT_INIT,
    INTUITIVE_CATEGORICAL_INPUT,
    INTUITIVE_DISTANCE_MODE,
    INTUITIVE_NON_MEMBERSHIP,
    INTUITIVE_NUMERIC_INPUT,
    INTUITIVE_NUMERIC_PREPROCESSING,
    INTUITIVE_SELECTION_METRIC,
    INTUITIVE_STOPPING_MODE,
    INTUITIVE_VIEW_WEIGHTED_DISTANCE_MODE,
    INTUITIVE_ZERO_PHI_POLICY,
)
from pipeline.fpmax import BIN_LABELS_BY_SIZE, DEFAULT_STRATEGIES

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_CONFIG_PATH = PROJECT_DIR / "configs" / "intuitive_protocols.json"
PRIMARY_PROTOCOL_ID = "mimvdec_intuitive_v1"
VIEW_WEIGHTED_PROTOCOL_ID = "mimvdec_intuitive_view_weighted_v1"
PROTOCOL_CONFIG_SCHEMA_VERSION = 2
WITHOUT_FFS_MODE = "intuitive_native_all_fpmax_features"
FFS_MODE = "intuitive_native_forward_selection"
VIEW_WEIGHTED_WITHOUT_FFS_MODE = "intuitive_view_weighted_native_all_fpmax_features"
VIEW_WEIGHTED_FFS_MODE = "intuitive_view_weighted_native_forward_selection"
PARAMETER_SEARCH = "two_stage"


@dataclass(frozen=True)
class IntuitiveProtocol:
    """Validated algorithm contract for one private Intuitive workflow."""

    protocol_id: str
    numeric_input: str
    numeric_preprocessing: str
    categorical_input: str
    initialization_strategy: str
    strict_initialization: bool
    non_membership: str
    zero_phi_policy: str
    distance_mode: str
    selection_metric: str
    stopping_mode: str
    max_iter: int
    empty_cluster_policy: str
    min_cluster_size: int
    without_ffs_mode: str
    ffs_mode: str
    parameter_search: str
    ffs_min_improvement: float
    fpmax_strategies: tuple[str, ...]
    fpmax_n_bins: tuple[int, ...]
    fpmax_min_supports: tuple[float, ...]
    mu_params: tuple[float, ...]
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    view_weight_alphas: tuple[float, ...]


def load_intuitive_protocol(
    config_path: Path,
    protocol_id: str,
) -> IntuitiveProtocol:
    """Load and validate one versioned Intuitive protocol contract."""

    safe_identifier = protocol_id.replace("_", "").replace("-", "")
    if not safe_identifier.isalnum() or Path(protocol_id).name != protocol_id:
        msg = f"Unsafe Intuitive protocol identifier: {protocol_id!r}."
        raise ValueError(msg)
    with config_path.open(encoding="utf-8") as file:
        registry = json.load(file)
    if registry.get("schema_version") != PROTOCOL_CONFIG_SCHEMA_VERSION:
        msg = f"Unsupported Intuitive protocol config schema: {config_path}."
        raise ValueError(msg)

    protocols = registry.get("protocols")
    if not isinstance(protocols, dict) or protocol_id not in protocols:
        available = sorted(protocols) if isinstance(protocols, dict) else []
        msg = f"Unknown Intuitive protocol {protocol_id!r}; available: {available}."
        raise ValueError(msg)
    raw_protocol = protocols[protocol_id]
    if not isinstance(raw_protocol, dict):
        msg = f"Intuitive protocol must be an object: {protocol_id}."
        raise ValueError(msg)

    expected_fields = {
        "numeric_input",
        "numeric_preprocessing",
        "categorical_input",
        "initialization_strategy",
        "strict_initialization",
        "non_membership",
        "zero_phi_policy",
        "distance_mode",
        "selection_metric",
        "stopping_mode",
        "max_iter",
        "empty_cluster_policy",
        "min_cluster_size",
        "without_ffs_mode",
        "ffs_mode",
        "parameter_search",
        "ffs_min_improvement",
        "fpmax_strategies",
        "fpmax_n_bins",
        "fpmax_min_supports",
        "mu_params",
        "gammas",
        "betas",
        "view_weight_alphas",
    }
    if set(raw_protocol) != expected_fields:
        missing = sorted(expected_fields - set(raw_protocol))
        unknown = sorted(set(raw_protocol) - expected_fields)
        msg = (
            f"Invalid fields for Intuitive protocol {protocol_id}: "
            f"missing={missing}, unknown={unknown}."
        )
        raise ValueError(msg)

    string_fields = expected_fields - {
        "strict_initialization",
        "max_iter",
        "min_cluster_size",
        "ffs_min_improvement",
        "fpmax_strategies",
        "fpmax_n_bins",
        "fpmax_min_supports",
        "mu_params",
        "gammas",
        "betas",
        "view_weight_alphas",
    }
    for field in string_fields:
        value = raw_protocol[field]
        if not isinstance(value, str) or not value:
            msg = f"Intuitive protocol field {field!r} must be a non-empty string."
            raise ValueError(msg)
    if not isinstance(raw_protocol["strict_initialization"], bool):
        msg = "Intuitive protocol strict_initialization must be boolean."
        raise ValueError(msg)
    for field in ("max_iter", "min_cluster_size"):
        value = raw_protocol[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"Intuitive protocol field {field!r} must be a positive integer."
            raise ValueError(msg)

    min_improvement = raw_protocol["ffs_min_improvement"]
    if (
        isinstance(min_improvement, bool)
        or not isinstance(min_improvement, (int, float))
        or not math.isfinite(min_improvement)
        or min_improvement < 0
    ):
        msg = "Intuitive protocol ffs_min_improvement must be finite and non-negative."
        raise ValueError(msg)

    strategies = raw_protocol["fpmax_strategies"]
    if (
        not isinstance(strategies, list)
        or not strategies
        or any(strategy not in DEFAULT_STRATEGIES for strategy in strategies)
        or len(set(strategies)) != len(strategies)
    ):
        msg = (
            "Intuitive protocol fpmax_strategies must be a non-empty unique subset "
            f"of {DEFAULT_STRATEGIES}."
        )
        raise ValueError(msg)

    n_bins_values = raw_protocol["fpmax_n_bins"]
    if (
        not isinstance(n_bins_values, list)
        or not n_bins_values
        or any(
            isinstance(n_bins, bool)
            or not isinstance(n_bins, int)
            or n_bins not in BIN_LABELS_BY_SIZE
            for n_bins in n_bins_values
        )
        or len(set(n_bins_values)) != len(n_bins_values)
    ):
        msg = (
            "Intuitive protocol fpmax_n_bins must be a non-empty unique subset "
            f"of {tuple(BIN_LABELS_BY_SIZE)}."
        )
        raise ValueError(msg)

    numeric_grids: dict[str, tuple[float, ...]] = {}
    for field in (
        "fpmax_min_supports",
        "mu_params",
        "gammas",
        "betas",
        "view_weight_alphas",
    ):
        values = raw_protocol[field]
        if not isinstance(values, list) or (
            field != "view_weight_alphas" and not values
        ):
            msg = f"Intuitive protocol field {field!r} must be a non-empty list."
            raise ValueError(msg)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            msg = f"Intuitive protocol field {field!r} must contain finite numbers."
            raise ValueError(msg)
        grid = tuple(float(value) for value in values)
        if len(set(grid)) != len(grid):
            msg = f"Intuitive protocol field {field!r} must contain unique values."
            raise ValueError(msg)
        numeric_grids[field] = grid
    if any(not 0 < value <= 1 for value in numeric_grids["fpmax_min_supports"]):
        msg = "Intuitive fpmax_min_supports must satisfy 0 < support <= 1."
        raise ValueError(msg)
    if any(not 0 <= value <= 1 for value in numeric_grids["mu_params"]):
        msg = "Intuitive mu_params must satisfy 0 <= mu <= 1."
        raise ValueError(msg)
    if any(not 0 <= value <= 1 for value in numeric_grids["gammas"]):
        msg = "Intuitive gammas must satisfy 0 <= gamma <= 1."
        raise ValueError(msg)
    if any(value <= 1 for value in numeric_grids["betas"]):
        msg = "Intuitive betas must be greater than 1."
        raise ValueError(msg)
    if any(not 0 <= value <= 1 for value in numeric_grids["view_weight_alphas"]):
        msg = "Intuitive view_weight_alphas must satisfy 0 <= alpha <= 1."
        raise ValueError(msg)

    protocol_values = dict(raw_protocol)
    protocol_values.update(numeric_grids)
    protocol_values.update(
        {
            "ffs_min_improvement": float(min_improvement),
            "fpmax_strategies": tuple(strategies),
            "fpmax_n_bins": tuple(n_bins_values),
        }
    )
    protocol = IntuitiveProtocol(protocol_id=protocol_id, **protocol_values)
    view_weighted = bool(protocol.view_weight_alphas)
    supported_values = {
        "numeric_input": INTUITIVE_NUMERIC_INPUT,
        "numeric_preprocessing": INTUITIVE_NUMERIC_PREPROCESSING,
        "categorical_input": INTUITIVE_CATEGORICAL_INPUT,
        "initialization_strategy": DEFAULT_INTUITIVE_INIT_STRATEGY,
        "strict_initialization": DEFAULT_INTUITIVE_STRICT_INIT,
        "non_membership": INTUITIVE_NON_MEMBERSHIP,
        "zero_phi_policy": INTUITIVE_ZERO_PHI_POLICY,
        "distance_mode": (
            INTUITIVE_VIEW_WEIGHTED_DISTANCE_MODE
            if view_weighted
            else INTUITIVE_DISTANCE_MODE
        ),
        "selection_metric": INTUITIVE_SELECTION_METRIC,
        "stopping_mode": INTUITIVE_STOPPING_MODE,
        "empty_cluster_policy": DEFAULT_INTUITIVE_EMPTY_CLUSTER_POLICY,
        "min_cluster_size": DEFAULT_INTUITIVE_MIN_CLUSTER_SIZE,
        "without_ffs_mode": (
            VIEW_WEIGHTED_WITHOUT_FFS_MODE if view_weighted else WITHOUT_FFS_MODE
        ),
        "ffs_mode": VIEW_WEIGHTED_FFS_MODE if view_weighted else FFS_MODE,
        "parameter_search": PARAMETER_SEARCH,
    }
    mismatches = {
        field: {"configured": getattr(protocol, field), "supported": supported}
        for field, supported in supported_values.items()
        if getattr(protocol, field) != supported
    }
    if mismatches:
        msg = (
            f"Unsupported settings for Intuitive protocol {protocol_id}: {mismatches}."
        )
        raise ValueError(msg)
    return protocol
