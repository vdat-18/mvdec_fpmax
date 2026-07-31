import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from utils import log_csv

from config import DEKM_DATASET_DIR, PUBLIC_BENCHMARK_OUTPUT_DIR
from pipeline.external_metrics import ExternalMetrics, compute_external_metrics
from pipeline.mvdec_contract import (
    FUSION_ENCODER_AVERAGE,
    VIEW2_ARCHITECTURE_ID,
    VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
)
from pipeline.mvdec_runs import (
    append_run_log,
    build_summary_frames,
    load_run_manifests,
    prepare_run_directory,
    resolve_run_paths,
    write_run_manifest,
)
from pipeline.public_artifacts import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    public_assignment_frame,
    write_public_artifact,
    write_public_assignments,
    write_public_frame,
)
from pipeline.public_data import load_dekm_public_dataset

ds_name = 'AIRPOLLUTION'
input_shape = 13
hidden_units = 10
n_clusters = 4
view1_filters = [500, 500, 2000]
view2_base_units = 64
pretrain_epochs = 200
batch_size = 256
UNLABELED_ASSIGNMENT_CHANGE_TOLERANCE = 0.01
PUBLIC_ASSIGNMENT_CHANGE_TOLERANCE = 0.001
assignment_change_tolerance = UNLABELED_ASSIGNMENT_CHANGE_TOLERANCE
# DEKM 2021 (Fig. 4) reports that the greedy last-direction update outperforms
# pulling every transformed dimension on MNIST. L2 and its trace-equivalent L3
# therefore remain diagnostics rather than optimized terms in the primary
# DEKM-consistent protocol.
DEFAULT_LAMBDA_KMEANS = 0.0
DEFAULT_LAMBDA_GREEDY = 1.0
lambda_reconstruction = 1.0
lambda_kmeans = DEFAULT_LAMBDA_KMEANS
lambda_orthonormal = 0.0
lambda_greedy = DEFAULT_LAMBDA_GREEDY
GREEDY_EIGEN_DIRECTIONS = ("largest", "smallest")
GREEDY_TARGET_MODES = ("selected_dimension_only", "frozen_snapshot")
DEFAULT_GREEDY_EIGEN_DIRECTION = "largest"
DEFAULT_GREEDY_TARGET_MODE = "frozen_snapshot"
PRIMARY_PROTOCOL_ID = "mvdec_dekm_consistent_v1"
PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID = "mvdec_dekm_consistent_public_v1"
PUBLIC_REPRODUCTION_PROTOCOL_ID = "mvdec_2025_public_reproduction_v1"
PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID = (
    "mvdec_2025_view2_encoder_bottleneck_v1"
)
PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID = "mvdec_2025_view2_direct_23_split_v1"
PUBLIC_REPRODUCTION_PROTOCOL_IDS = frozenset(
    {
        PUBLIC_REPRODUCTION_PROTOCOL_ID,
        PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID,
        PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID,
    }
)
PUBLIC_PROTOCOL_IDS = frozenset(
    {PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID, *PUBLIC_REPRODUCTION_PROTOCOL_IDS}
)
CUSTOM_PROTOCOL_ID = "custom"
PROTOCOL_IDS = (
    PRIMARY_PROTOCOL_ID,
    PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID,
    PUBLIC_REPRODUCTION_PROTOCOL_ID,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID,
    PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID,
    CUSTOM_PROTOCOL_ID,
)
EIGENVALUE_ORDER = "ascending"
L4_REDUCTION = "sum_squared_dimensions_then_mean_batch"
RELEASE_L4_REDUCTION = "mean_squared_dimensions_per_sample_sum_batch_gradient"
KMEANS_N_INIT = 100
KMEANS_REFRESH_POLICY = "one_epoch"
REFINEMENT_BATCHING_POLICY = "balanced_shuffled_each_epoch"
FIXED_KMEANS_N_INIT_POLICY = "fixed_100"
RELEASE_KMEANS_N_INIT_POLICY = "initial_100_then_twice_previous_n_iter"
RELEASE_KMEANS_REFRESH_INTERVAL = 10
RELEASE_MAX_TRAINING_STEPS = 14_000
RELEASE_PRETRAIN_SHUFFLE_BUFFER = 8_000
RELEASE_KMEANS_REFRESH_POLICY = "fixed_10_updates"
RELEASE_REFINEMENT_BATCHING_POLICY = "sequential_release_order"
JOINT_REFINEMENT_OBJECTIVE = "joint_reconstruction_plus_greedy"
RELEASE_REFINEMENT_OBJECTIVE = "release_reconstruction_plus_greedy_mse"
SUM_SQUARED_PRETRAIN_REDUCTION = "sum_squared_dimensions_per_sample"
RELEASE_PRETRAIN_REDUCTION = "mean_squared_dimensions_per_sample"
DETERMINISTIC_RUNTIME_POLICY = "keras_seeded_tf_deterministic_v1"
MAX_REFINEMENT_EPOCHS = 1400
DEFAULT_PROGRESS_INTERVAL = 25
TRAINING_STOP_REASONS = ("converged_assignment", "max_epochs_reached")
PUBLIC_DATASETS = ("REUTERS", "20NEWS", "RCV1")
UNLABELED_SCALING_METHODS = ("minmax", "standard", "none")
UNLABELED_DATASETS = {
    'AIRPOLLUTION': {
        'csv_path': 'data/preprocessed_data/data_demvk.csv',
        'input_shape': 13,
        'hidden_units': 10,
        'view1_filters': [500, 500, 2000],
        'view2_base_units': 64,
        'n_clusters': 4,
        'scaling_method': 'minmax',
    },
    'TIKI': {
        'csv_path': 'data/preprocessed_data/tiki_preprocessed.csv',
        'input_shape': 7,
        'hidden_units': 5,
        'view1_filters': [250, 250, 1000],
        'view2_base_units': 32,
        'n_clusters': 5,
        'scaling_method': 'none',
    },
}


@dataclass(frozen=True)
class MvdecProtocol:
    """Resolved immutable identity for one MvDEC training configuration."""

    protocol_id: str
    claim_scope: str
    architecture_source: str
    view2_architecture_id: str
    fusion_contract: str
    refinement_source: str
    reconstruction_weight: float
    kmeans_weight: float
    scatter_trace_weight: float
    greedy_weight: float
    eigenvalue_order: str
    greedy_eigen_direction: str
    greedy_target_mode: str
    l4_reduction: str
    refinement_objective: str
    kmeans_refresh_policy: str
    kmeans_refresh_interval: int | None
    refinement_batching_policy: str
    max_training_steps: int | None
    kmeans_n_init_policy: str
    pretrain_loss_reduction: str
    pretrain_shuffle_buffer: int | None

    def manifest_contract(self, tolerance: float) -> dict[str, object]:
        """Return the explicit paper and optimization contract for artifacts."""

        contract = {
            "protocol_id": self.protocol_id,
            "claim_scope": self.claim_scope,
            "architecture_source": self.architecture_source,
            "architecture": {
                "view2": self.view2_architecture_id,
            },
            "refinement_source": self.refinement_source,
            "objective": {
                "name": final_training_objective(self),
                "loss_terms": {
                    "L1_reconstruction": {
                        "weight": self.reconstruction_weight,
                        "optimized": self.reconstruction_weight > 0,
                    },
                    'L2_kmeans': {
                        'weight': self.kmeans_weight,
                        'optimized': self.kmeans_weight > 0,
                    },
                    'L3_scatter_trace': {
                        'weight': self.scatter_trace_weight,
                        'optimized': self.scatter_trace_weight > 0,
                        'mathematically_equivalent_to_L2': True,
                    },
                    'L4_greedy': {
                        'weight': self.greedy_weight,
                        'optimized': self.greedy_weight > 0,
                        'reduction': self.l4_reduction,
                    },
                },
            },
            'eigen': {
                'order': self.eigenvalue_order,
                'direction': self.greedy_eigen_direction,
                'rationale': (
                    'DEKM 2021 sorts ascending and uses the largest/last '
                    'least-informative direction.'
                ),
            },
            'greedy_target': {
                'mode': self.greedy_target_mode,
                'rationale': (
                    'Frozen transformed-embedding target used by the released '
                    'DEKM 2021 implementation.'
                ),
            },
            'stopping': {
                'assignment_change_tolerance': float(tolerance),
                'policy': (
                    '0.001 for public DEKM benchmarks; 0.01 by design for '
                    'unlabeled private case studies.'
                ),
            },
        }
        if self.fusion_contract != FUSION_ENCODER_AVERAGE:
            contract["architecture"]["fusion"] = self.fusion_contract
        if self.protocol_id in PUBLIC_PROTOCOL_IDS:
            contract["schedule"] = {
                "scope": "public_datasets_only",
                "pretraining": {
                    "epochs": 200,
                    "batch_size": 256,
                    "loss_reduction": self.pretrain_loss_reduction,
                    "shuffle_buffer": self.pretrain_shuffle_buffer,
                },
                "refinement": {
                    "objective": self.refinement_objective,
                    "batch_size": 256,
                    "batching_policy": self.refinement_batching_policy,
                    "kmeans_refresh_policy": self.kmeans_refresh_policy,
                    "update_interval": self.kmeans_refresh_interval,
                    "max_training_steps": self.max_training_steps,
                    "kmeans_n_init_policy": self.kmeans_n_init_policy,
                },
                "final_evaluation": {
                    "representations": ["view1", "view2", "fused"],
                    "clustering": "independent_kmeans_n_init_100",
                    "metrics": ["acc", "nmi"],
                    "ground_truth_usage": "final_evaluation_only",
                },
            }
        return contract


PRIMARY_MVDEC_PROTOCOL = MvdecProtocol(
    protocol_id=PRIMARY_PROTOCOL_ID,
    claim_scope='MvDEC architecture with DEKM-2021-consistent greedy refinement',
    architecture_source='MvDEC 2025 multi-view encoder-latent fusion',
    view2_architecture_id=VIEW2_ARCHITECTURE_ID,
    fusion_contract=FUSION_ENCODER_AVERAGE,
    refinement_source='DEKM 2021 Algorithm 1 and released implementation',
    reconstruction_weight=1.0,
    kmeans_weight=DEFAULT_LAMBDA_KMEANS,
    scatter_trace_weight=0.0,
    greedy_weight=DEFAULT_LAMBDA_GREEDY,
    eigenvalue_order=EIGENVALUE_ORDER,
    greedy_eigen_direction=DEFAULT_GREEDY_EIGEN_DIRECTION,
    greedy_target_mode=DEFAULT_GREEDY_TARGET_MODE,
    l4_reduction=L4_REDUCTION,
    refinement_objective=JOINT_REFINEMENT_OBJECTIVE,
    kmeans_refresh_policy=KMEANS_REFRESH_POLICY,
    kmeans_refresh_interval=None,
    refinement_batching_policy=REFINEMENT_BATCHING_POLICY,
    max_training_steps=None,
    kmeans_n_init_policy=FIXED_KMEANS_N_INIT_POLICY,
    pretrain_loss_reduction=SUM_SQUARED_PRETRAIN_REDUCTION,
    pretrain_shuffle_buffer=None,
)


PUBLIC_REPRODUCTION_PROTOCOL = MvdecProtocol(
    protocol_id=PUBLIC_REPRODUCTION_PROTOCOL_ID,
    claim_scope=(
        "MvDEC 2025 public architecture with MvDEC L1 reconstruction and "
        "historical released DEKM-style L4 optimization; reproduction contract, "
        "not an exact-paper claim"
    ),
    architecture_source="MvDEC 2025 multi-view encoder-latent fusion",
    view2_architecture_id=VIEW2_ARCHITECTURE_ID,
    fusion_contract=FUSION_ENCODER_AVERAGE,
    refinement_source=(
        "MvDEC L1 reconstruction with released DEKM L4 training behavior"
    ),
    reconstruction_weight=1.0,
    kmeans_weight=0.0,
    scatter_trace_weight=0.0,
    greedy_weight=1.0,
    eigenvalue_order=EIGENVALUE_ORDER,
    greedy_eigen_direction=DEFAULT_GREEDY_EIGEN_DIRECTION,
    greedy_target_mode=DEFAULT_GREEDY_TARGET_MODE,
    l4_reduction=RELEASE_L4_REDUCTION,
    refinement_objective=RELEASE_REFINEMENT_OBJECTIVE,
    kmeans_refresh_policy=RELEASE_KMEANS_REFRESH_POLICY,
    kmeans_refresh_interval=RELEASE_KMEANS_REFRESH_INTERVAL,
    refinement_batching_policy=RELEASE_REFINEMENT_BATCHING_POLICY,
    max_training_steps=RELEASE_MAX_TRAINING_STEPS,
    kmeans_n_init_policy=RELEASE_KMEANS_N_INIT_POLICY,
    pretrain_loss_reduction=RELEASE_PRETRAIN_REDUCTION,
    pretrain_shuffle_buffer=RELEASE_PRETRAIN_SHUFFLE_BUFFER,
)

PUBLIC_DEKM_CONSISTENT_PROTOCOL = replace(
    PRIMARY_MVDEC_PROTOCOL,
    protocol_id=PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID,
    claim_scope=(
        "DEKM-2021-consistent MvDEC objective adapted to the bounded public "
        "benchmark schedule; diagnostic protocol, not an exact-paper claim"
    ),
    refinement_source=(
        "DEKM 2021 objective and batching with the bounded MvDEC public schedule"
    ),
    kmeans_refresh_policy=RELEASE_KMEANS_REFRESH_POLICY,
    kmeans_refresh_interval=RELEASE_KMEANS_REFRESH_INTERVAL,
    max_training_steps=RELEASE_MAX_TRAINING_STEPS,
)

PUBLIC_ENCODER_BOTTLENECK_PROTOCOL = replace(
    PUBLIC_REPRODUCTION_PROTOCOL,
    protocol_id=PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID,
    claim_scope=(
        "MvDEC 2025 public-reproduction diagnostic with the View2 latent placed "
        "at the encoder bottleneck; architecture ablation, not an exact-paper claim"
    ),
    architecture_source=(
        "MvDEC 2025 dense U-Net with encoder-bottleneck latent and skip decoder"
    ),
    view2_architecture_id=VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
)

PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL = replace(
    PUBLIC_REPRODUCTION_PROTOCOL,
    protocol_id=PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID,
    claim_scope=(
        "MvDEC 2025 public-reproduction diagnostic with the Fig. 2 View2 "
        "direct latent-reconstruction head; architecture ablation, not an "
        "exact-paper claim"
    ),
    architecture_source=(
        "MvDEC 2025 dense U-Net with a post-skip direct joint linear head"
    ),
    view2_architecture_id=VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
)

PUBLIC_PROTOCOLS = {
    PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID: PUBLIC_DEKM_CONSISTENT_PROTOCOL,
    PUBLIC_REPRODUCTION_PROTOCOL_ID: PUBLIC_REPRODUCTION_PROTOCOL,
    PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID: PUBLIC_ENCODER_BOTTLENECK_PROTOCOL,
    PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID: PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL,
}


def resolve_assignment_change_tolerance(dataset_name, override=None):
    """Return an explicit valid tolerance for one dataset protocol."""

    if override is None:
        return (
            UNLABELED_ASSIGNMENT_CHANGE_TOLERANCE
            if dataset_name in UNLABELED_DATASETS
            else PUBLIC_ASSIGNMENT_CHANGE_TOLERANCE
        )
    tolerance = float(override)
    if not 0 < tolerance < 1:
        raise ValueError(
            '--assignment-change-tolerance must be strictly between 0 and 1.'
        )
    return tolerance


def validate_protocol_assignment_change_tolerance(
    protocol: MvdecProtocol,
    dataset_name: str,
    tolerance: float,
) -> float:
    """Require custom protocol identity for stopping-tolerance ablations."""

    expected = resolve_assignment_change_tolerance(dataset_name)
    if (
        protocol.protocol_id
        in {
            PRIMARY_PROTOCOL_ID,
            *PUBLIC_PROTOCOL_IDS,
        }
        and tolerance != expected
    ):
        raise ValueError(
            f"{protocol.protocol_id} fixes assignment-change tolerance at "
            f"{expected:g} for {dataset_name}. Use --protocol custom for "
            "tolerance ablations."
        )
    return tolerance


def validate_protocol_dataset_scope(
    protocol: MvdecProtocol,
    dataset_name: str,
) -> None:
    """Reject use of the public reproduction contract on private datasets."""

    if (
        protocol.protocol_id in PUBLIC_PROTOCOL_IDS
        and dataset_name not in PUBLIC_DATASETS
    ):
        raise ValueError(
            f"{protocol.protocol_id} supports only "
            f"{', '.join(PUBLIC_DATASETS)}; got {dataset_name!r}."
        )


def validate_protocol_max_refinement_epochs(
    protocol: MvdecProtocol,
    max_refinement_epochs: int,
) -> int:
    """Reject an epoch override that cannot alter the fixed release budget."""

    value = validate_max_refinement_epochs(max_refinement_epochs)
    if (
        protocol.protocol_id in PUBLIC_PROTOCOL_IDS
        and value != MAX_REFINEMENT_EPOCHS
    ):
        raise ValueError(
            f"{protocol.protocol_id} fixes refinement at "
            f"{RELEASE_MAX_TRAINING_STEPS} updates. "
            "--max-refinement-epochs applies only to the primary/custom protocols."
        )
    return value


def view_output_width():
    return hidden_units + input_shape


def view_output_layout():
    return f'eq5_compatible_{hidden_units}_plus_{input_shape}'


def final_training_objective(protocol: MvdecProtocol) -> str:
    """Return an objective name that does not overclaim paper fidelity."""

    if protocol.protocol_id in {
        PRIMARY_PROTOCOL_ID,
        PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID,
    }:
        return "mvdec_dekm_consistent_l1_reconstruction_plus_l4_greedy"
    if protocol.protocol_id in PUBLIC_REPRODUCTION_PROTOCOL_IDS:
        return (
            "mvdec_2025_public_reproduction_release_"
            "l1_reconstruction_plus_l4_greedy_mse"
        )
    parts = ["reconstruction"]
    if protocol.kmeans_weight > 0:
        parts.append('kmeans')
    if protocol.scatter_trace_weight > 0:
        parts.append('scatter_trace')
    if protocol.greedy_weight > 0:
        parts.append('greedy')
    return f'mvdec_custom_{"_".join(parts)}'


def resolve_mvdec_protocol(
    protocol_id: str,
    kmeans_weight: float,
    greedy_weight: float,
    eigen_direction: str,
    target_mode: str,
) -> MvdecProtocol:
    """Resolve and validate a named primary protocol or explicit ablation."""

    if protocol_id not in PROTOCOL_IDS:
        raise ValueError(f'Unsupported MvDEC protocol: {protocol_id!r}.')
    if kmeans_weight < 0 or greedy_weight < 0:
        raise ValueError('MvDEC loss weights must be non-negative.')
    greedy_eigen_index(eigen_direction)
    validate_greedy_target_mode(target_mode)
    if protocol_id in PUBLIC_PROTOCOL_IDS:
        public_protocol = PUBLIC_PROTOCOLS[protocol_id]
        if any(
            (
                kmeans_weight != public_protocol.kmeans_weight,
                greedy_weight != public_protocol.greedy_weight,
                eigen_direction != public_protocol.greedy_eigen_direction,
                target_mode != public_protocol.greedy_target_mode,
            )
        ):
            raise ValueError(
                f"{protocol_id} is immutable. Use "
                "--protocol custom for loss, eigen-direction, or target-mode "
                "ablations."
            )
        return public_protocol

    resolved = MvdecProtocol(
        protocol_id=protocol_id,
        claim_scope=(
            PRIMARY_MVDEC_PROTOCOL.claim_scope
            if protocol_id == PRIMARY_PROTOCOL_ID
            else 'Explicit custom ablation; not a faithful paper-replication claim'
        ),
        architecture_source=PRIMARY_MVDEC_PROTOCOL.architecture_source,
        view2_architecture_id=PRIMARY_MVDEC_PROTOCOL.view2_architecture_id,
        fusion_contract=PRIMARY_MVDEC_PROTOCOL.fusion_contract,
        refinement_source=(
            PRIMARY_MVDEC_PROTOCOL.refinement_source
            if protocol_id == PRIMARY_PROTOCOL_ID
            else 'User-configured MvDEC ablation'
        ),
        reconstruction_weight=1.0,
        kmeans_weight=float(kmeans_weight),
        scatter_trace_weight=0.0,
        greedy_weight=float(greedy_weight),
        eigenvalue_order=EIGENVALUE_ORDER,
        greedy_eigen_direction=eigen_direction,
        greedy_target_mode=target_mode,
        l4_reduction=L4_REDUCTION,
        refinement_objective=JOINT_REFINEMENT_OBJECTIVE,
        kmeans_refresh_policy=KMEANS_REFRESH_POLICY,
        kmeans_refresh_interval=None,
        refinement_batching_policy=REFINEMENT_BATCHING_POLICY,
        max_training_steps=None,
        kmeans_n_init_policy=FIXED_KMEANS_N_INIT_POLICY,
        pretrain_loss_reduction=SUM_SQUARED_PRETRAIN_REDUCTION,
        pretrain_shuffle_buffer=None,
    )
    if protocol_id == PRIMARY_PROTOCOL_ID and resolved != PRIMARY_MVDEC_PROTOCOL:
        raise ValueError(
            f'{PRIMARY_PROTOCOL_ID} is immutable. Use --protocol custom for '
            'loss, eigen-direction, or target-mode ablations.'
        )
    return resolved


def protocol_contract_sha256(contract: dict[str, object]) -> str:
    """Hash the canonical resolved protocol contract stored in an artifact."""

    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol_method_name(protocol: MvdecProtocol) -> str:
    """Return the human-readable method label for tables and artifacts."""

    if protocol.protocol_id == PRIMARY_PROTOCOL_ID:
        return "MvDEC-DEKM-consistent"
    if protocol.protocol_id == PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID:
        return "MvDEC-DEKM-consistent-public"
    if protocol.protocol_id == PUBLIC_ENCODER_BOTTLENECK_PROTOCOL_ID:
        return "MvDEC-2025-View2-encoder-bottleneck-ablation"
    if protocol.protocol_id == PUBLIC_DIRECT_JOINT_HEAD_PROTOCOL_ID:
        return "MvDEC-2025-View2-direct-joint-head-ablation"
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        return "MvDEC-2025-public-reproduction"
    return "MvDEC custom ablation"


def resolved_run_config(
    protocol: MvdecProtocol,
    dataset_name: str,
    tolerance: float,
    max_refinement_epochs: int,
    source_sha256: str | None,
    feature_columns: list[str],
    preprocessing_metadata: dict[str, object] | None,
) -> dict[str, object]:
    """Return the full seed-independent configuration used for run identity."""

    config = {
        "dataset": dataset_name,
        "method": protocol_method_name(protocol),
        "protocol_contract": protocol.manifest_contract(tolerance),
        "source_sha256": source_sha256,
        "feature_columns": list(feature_columns),
        "preprocessing": preprocessing_metadata,
        "n_clusters": int(n_clusters),
        "input_shape": int(input_shape),
        "hidden_units": int(hidden_units),
        "view1_filters": list(view1_filters),
        "view2_base_units": int(view2_base_units),
        "view2_architecture_id": protocol.view2_architecture_id,
        "pretrain_epochs": int(pretrain_epochs),
        "batch_size": int(batch_size),
        "kmeans_n_init": int(KMEANS_N_INIT),
        "kmeans_refresh_policy": protocol.kmeans_refresh_policy,
        "refinement_batching_policy": protocol.refinement_batching_policy,
        "deterministic_runtime_policy": DETERMINISTIC_RUNTIME_POLICY,
        "max_refinement_epochs": int(max_refinement_epochs),
        "assignment_change_tolerance": float(tolerance),
        "l4_reduction": protocol.l4_reduction,
    }
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        config.update(
            {
                "pretrain_loss_reduction": protocol.pretrain_loss_reduction,
                "pretrain_shuffle_buffer": protocol.pretrain_shuffle_buffer,
                "refinement_objective": protocol.refinement_objective,
                "update_interval": protocol.kmeans_refresh_interval,
                "max_training_steps": protocol.max_training_steps,
                "kmeans_n_init_policy": protocol.kmeans_n_init_policy,
            }
        )
    if protocol.fusion_contract != FUSION_ENCODER_AVERAGE:
        config["fusion_contract"] = protocol.fusion_contract
    return config


def latent_embedding(view_output):
    return view_output[:, :hidden_units]


def reconstruction_output(view_output):
    return view_output[:, -input_shape:]


def fused_latent_embedding(
    view1_output,
    view2_output,
    fusion_contract=FUSION_ENCODER_AVERAGE,
):
    """Fuse two latent tensors according to the immutable protocol contract."""

    h_view1 = latent_embedding(view1_output)
    h_view2 = latent_embedding(view2_output)
    if fusion_contract == FUSION_ENCODER_AVERAGE:
        return (h_view1 + h_view2) / 2
    raise ValueError(f"Unsupported MvDEC fusion contract: {fusion_contract!r}.")


def greedy_eigen_index(direction):
    if direction == 'largest':
        return -1
    if direction == 'smallest':
        return 0
    raise ValueError(f'Unsupported greedy eigen direction: {direction!r}.')


def validate_greedy_target_mode(target_mode):
    if target_mode not in GREEDY_TARGET_MODES:
        raise ValueError(f'Unsupported greedy target mode: {target_mode!r}.')
    return target_mode


def build_greedy_target(
    transformed_embeddings,
    transformed_centroids,
    assignments,
    eigen_index,
):
    target = np.array(transformed_embeddings, copy=True)
    target[:, eigen_index] = transformed_centroids[assignments, eigen_index]
    return target


def loss_ablation_tag(kmeans_weight, greedy_weight):
    if (
        kmeans_weight == DEFAULT_LAMBDA_KMEANS
        and greedy_weight == DEFAULT_LAMBDA_GREEDY
    ):
        return ''
    return f'_l2w{kmeans_weight:g}_l4w{greedy_weight:g}'


def configure_deterministic_runtime(seed: int | None) -> dict[str, object]:
    """Reset Keras state and seed every RNG used by one isolated run."""

    if seed is None:
        raise ValueError('Deterministic MvDEC runs require an explicit seed.')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return {
        'policy': DETERMINISTIC_RUNTIME_POLICY,
        'seed': int(seed),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'keras_session_reset': True,
        'tensorflow_op_determinism': True,
        'tf_data_deterministic': True,
    }


def make_pretraining_dataset(
    x: np.ndarray,
    random_seed: int,
    protocol: MvdecProtocol = PRIMARY_MVDEC_PROTOCOL,
) -> tf.data.Dataset:
    """Build a reproducibly shuffled autoencoder pretraining dataset."""

    options = tf.data.Options()
    options.deterministic = True
    shuffle_buffer = protocol.pretrain_shuffle_buffer or len(x)
    return (
        tf.data.Dataset.from_tensor_slices((x, x))
        .shuffle(
            buffer_size=min(shuffle_buffer, len(x)),
            seed=random_seed,
            reshuffle_each_iteration=True,
        )
        .batch(pretrain_batch_size)
        .with_options(options)
    )


def make_kmeans(random_seed, n_init=KMEANS_N_INIT):
    """Build one seeded K-means estimator with an explicit restart count."""

    if n_init < 1:
        raise ValueError("K-Means n_init must be positive.")
    return KMeans(
        n_clusters=n_clusters,
        n_init=int(n_init),
        random_state=random_seed,
    )


def next_kmeans_n_init(
    protocol: MvdecProtocol,
    fitted_kmeans: KMeans,
) -> int:
    """Resolve the restart count for the next refinement checkpoint."""

    if protocol.kmeans_n_init_policy == FIXED_KMEANS_N_INIT_POLICY:
        return KMEANS_N_INIT
    if protocol.kmeans_n_init_policy == RELEASE_KMEANS_N_INIT_POLICY:
        return int(fitted_kmeans.n_iter_ * 2)
    raise ValueError(
        f"Unsupported K-Means n_init policy: {protocol.kmeans_n_init_policy!r}."
    )


def count_aligned_assignment_changes(previous_labels, current_labels):
    previous_labels = np.asarray(previous_labels)
    current_labels = np.asarray(current_labels)
    if previous_labels.shape != current_labels.shape:
        raise ValueError('Assignment arrays must have the same shape.')
    if np.any(previous_labels < 0):
        return int(len(current_labels))

    overlap = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for current_cluster, previous_cluster in zip(
        current_labels,
        previous_labels,
        strict=False,
    ):
        overlap[current_cluster, previous_cluster] += 1
    current_cluster_ids, previous_cluster_ids = linear_sum_assignment(-overlap)
    aligned_previous = previous_labels.copy()
    for current_cluster, previous_cluster in zip(
        current_cluster_ids,
        previous_cluster_ids,
        strict=False,
    ):
        aligned_previous[previous_labels == previous_cluster] = current_cluster
    return int(np.sum(current_labels != aligned_previous))


def number_of_batches(n_samples: int, current_batch_size: int) -> int:
    """Return the closest positive batch count to the target batch size."""

    if n_samples < 1 or current_batch_size < 1:
        raise ValueError('n_samples and batch_size must be positive.')
    return max(1, (n_samples + current_batch_size // 2) // current_batch_size)


@dataclass(frozen=True)
class RefinementSchedule:
    """Resolved update budget and batching cadence for one protocol run."""

    batches_per_epoch: int
    kmeans_refresh_interval: int
    max_training_steps: int


def resolve_refinement_schedule(
    protocol: MvdecProtocol,
    n_samples: int,
    current_batch_size: int,
    max_refinement_epochs: int,
) -> RefinementSchedule:
    """Resolve the immutable protocol schedule without changing old behavior."""

    max_refinement_epochs = validate_protocol_max_refinement_epochs(
        protocol,
        max_refinement_epochs,
    )
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        batches_per_epoch = (
            number_of_batches(n_samples, current_batch_size)
            if protocol.refinement_batching_policy == REFINEMENT_BATCHING_POLICY
            else (n_samples + current_batch_size - 1) // current_batch_size
        )
        return RefinementSchedule(
            batches_per_epoch=batches_per_epoch,
            kmeans_refresh_interval=int(protocol.kmeans_refresh_interval),
            max_training_steps=int(protocol.max_training_steps),
        )
    batches_per_epoch = number_of_batches(n_samples, current_batch_size)
    return RefinementSchedule(
        batches_per_epoch=batches_per_epoch,
        kmeans_refresh_interval=batches_per_epoch,
        max_training_steps=max_refinement_epochs * batches_per_epoch,
    )


def validate_max_refinement_epochs(max_refinement_epochs):
    if max_refinement_epochs < 1:
        raise ValueError('max_refinement_epochs must be at least 1.')
    return int(max_refinement_epochs)


def validate_progress_interval(progress_interval: int) -> int:
    """Return a valid positive interval for concise console heartbeats."""

    if progress_interval < 1:
        raise ValueError('progress_interval must be at least 1.')
    return int(progress_interval)


def should_log_progress(completed: int, total: int, interval: int) -> bool:
    """Log the first, final, and every configured progress interval."""

    interval = validate_progress_interval(interval)
    return completed == 1 or completed == total or completed % interval == 0


def epoch_batch_bounds(
    n_samples: int,
    current_batch_size: int,
) -> list[tuple[int, int]]:
    """Split an epoch into balanced contiguous bounds without dropping rows."""

    n_batches = number_of_batches(n_samples, current_batch_size)
    base_size, larger_batches = divmod(n_samples, n_batches)
    bounds = []
    start = 0
    for batch_index in range(n_batches):
        size = base_size + int(batch_index < larger_batches)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def epoch_batch_indices(
    n_samples: int,
    current_batch_size: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return one deterministic shuffled epoch of balanced mini-batches."""

    shuffled = rng.permutation(n_samples)
    return [
        shuffled[start:end]
        for start, end in epoch_batch_bounds(n_samples, current_batch_size)
    ]


def sequential_release_batch_indices(
    n_samples: int,
    current_batch_size: int,
    training_step: int,
) -> np.ndarray:
    """Return the historical contiguous batch, including the final remainder."""

    if n_samples < 1 or current_batch_size < 1 or training_step < 0:
        raise ValueError("Batch dimensions must be positive and step non-negative.")
    batches_per_epoch = (n_samples + current_batch_size - 1) // current_batch_size
    batch_index = training_step % batches_per_epoch
    start = batch_index * current_batch_size
    end = min(start + current_batch_size, n_samples)
    return np.arange(start, end)


def is_refinement_epoch_end(training_step, batches_per_epoch):
    return (training_step + 1) % batches_per_epoch == 0


def _file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as file:
        if path.suffix.lower() in {'.csv', '.tsv', '.txt'}:
            for line in file:
                digest.update(line.replace(b'\r\n', b'\n'))
        else:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def preprocess_unlabeled_features(df, source_path, scaling_method='none'):
    raw_x = df.to_numpy(dtype=np.float64)
    if not np.isfinite(raw_x).all():
        raise ValueError('Unlabeled input contains NaN or infinite values.')

    data_min = raw_x.min(axis=0)
    data_max = raw_x.max(axis=0)
    data_range = data_max - data_min
    constant_mask = data_range == 0

    if scaling_method == 'minmax':
        safe_range = np.where(constant_mask, 1.0, data_range)
        x = np.clip((raw_x - data_min) / safe_range, 0.0, 1.0)
        x[:, constant_mask] = 0.0
        feature_range = [0.0, 1.0]
        assumption = (
            'Column-wise Min-Max scaling selected for reproducibility because '
            'the MvDEC 2025 paper presents the 13 air-pollution features on a '
            '[0, 1] scale but does not publish the fitted scaler.'
        )
    elif scaling_method == 'standard':
        data_mean = raw_x.mean(axis=0)
        data_std = raw_x.std(axis=0)
        safe_std = np.where(constant_mask, 1.0, data_std)
        x = (raw_x - data_mean) / safe_std
        x[:, constant_mask] = 0.0
        feature_range = None
        assumption = (
            'Column-wise standard scaling is applied as a sensitivity '
            'experiment using population mean and standard deviation.'
        )
    elif scaling_method == 'none':
        x = raw_x.copy()
        feature_range = None
        assumption = 'No feature scaling is applied for this dataset.'
    else:
        raise ValueError(f'Unsupported scaling method: {scaling_method!r}.')

    source_path = Path(source_path)
    metadata = {
        'method': scaling_method,
        'feature_range': feature_range,
        'feature_columns': list(df.columns),
        'data_min': data_min.tolist(),
        'data_max': data_max.tolist(),
        'constant_features': [
            str(df.columns[index])
            for index in np.flatnonzero(constant_mask)
        ],
        'source_path': source_path.as_posix(),
        'source_sha256': _file_sha256(source_path),
        'assumption': assumption,
    }
    if scaling_method == 'standard':
        metadata['data_mean'] = data_mean.tolist()
        metadata['data_std'] = data_std.tolist()
    return raw_x.astype(np.float32), x.astype(np.float32), metadata


def resolve_unlabeled_scaling_method(dataset_name, override=None):
    """Return the configured or explicitly requested scaling method."""

    if dataset_name not in UNLABELED_DATASETS:
        raise ValueError(f'Unsupported unlabeled dataset: {dataset_name!r}.')
    scaling_method = (
        UNLABELED_DATASETS[dataset_name]['scaling_method']
        if override is None
        else override
    )
    if scaling_method not in UNLABELED_SCALING_METHODS:
        raise ValueError(f'Unsupported scaling method: {scaling_method!r}.')
    return scaling_method


def _preprocessing_input_space(
    preprocessing_metadata: dict[str, object] | None,
) -> str:
    """Return the auditable input-space label for training logs."""

    if preprocessing_metadata is None:
        return 'x'
    scaling_method = preprocessing_metadata.get('method')
    if scaling_method == 'none':
        return 'x_raw'
    if scaling_method in UNLABELED_SCALING_METHODS:
        return f'x_{scaling_method}'
    raise ValueError(f'Unsupported preprocessing metadata: {scaling_method!r}.')


def get_x_airpollution(
    dir_path=r'data/preprocessed_data/',
    log_print=True,
    shuffle_seed=None,
    scaling_method='minmax',
):
    return get_x_unlabeled_csv(
        Path(dir_path) / 'data_demvk.csv',
        log_print=log_print,
        shuffle_seed=shuffle_seed,
        scaling_method=scaling_method,
    )


def get_x_unlabeled_csv(
    csv_path,
    log_print=True,
    shuffle_seed=None,
    scaling_method='none',
    include_preprocessing=False,
):
    df = pd.read_csv(csv_path)
    raw_x, x, preprocessing_metadata = preprocess_unlabeled_features(
        df,
        source_path=csv_path,
        scaling_method=scaling_method,
    )
    if shuffle_seed is None:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    raw_x = raw_x[idx]
    if log_print:
        print(ds_name)
        print(
            f'preprocessing:{scaling_method}; '
            f'source_sha256:{preprocessing_metadata["source_sha256"][:12]}'
        )
    result = (x, idx, list(df.columns))
    if include_preprocessing:
        return (*result, raw_x, preprocessing_metadata)
    return result


def _restore_original_order(values, orig_idx):
    ordered = np.empty_like(values)
    ordered[orig_idx] = values
    return ordered


def save_airpollution_mvdec_artifact(
    artifact_path,
    h_view1,
    h_view2,
    h_fused,
    labels,
    score,
    iteration,
    orig_idx,
    feature_columns,
    random_seed,
    greedy_eigen_direction,
    greedy_target_mode,
    batches_per_epoch,
    max_refinement_epochs,
    stop_reason,
    refinement_epochs_completed,
    preprocessing_metadata,
    true_labels=None,
    source_sha256=None,
    external_metrics=None,
    representation_external_metrics=None,
    training_steps_completed=None,
    protocol=PRIMARY_MVDEC_PROTOCOL,
    run_id=None,
    config_hash=None,
):
    greedy_eigen_index(greedy_eigen_direction)
    validate_greedy_target_mode(greedy_target_mode)
    max_refinement_epochs = validate_protocol_max_refinement_epochs(
        protocol,
        max_refinement_epochs,
    )
    schedule = resolve_refinement_schedule(
        protocol,
        len(h_fused),
        batch_size,
        max_refinement_epochs,
    )
    if batches_per_epoch != schedule.batches_per_epoch:
        raise ValueError("MvDEC artifact batch count does not match its protocol.")
    if stop_reason not in TRAINING_STOP_REASONS:
        raise ValueError(f'Unsupported training stop reason: {stop_reason!r}.')
    greedy_eigen_position = (
        h_fused.shape[1] - 1 if greedy_eigen_direction == 'largest' else 0
    )
    if any(
        (
            protocol.greedy_eigen_direction != greedy_eigen_direction,
            protocol.greedy_target_mode != greedy_target_mode,
            protocol.kmeans_weight != lambda_kmeans,
            protocol.greedy_weight != lambda_greedy,
        )
    ):
        raise ValueError('MvDEC artifact settings do not match its protocol contract.')
    protocol_contract = protocol.manifest_contract(assignment_change_tolerance)
    protocol_sha256 = protocol_contract_sha256(protocol_contract)
    training_objective = final_training_objective(protocol)
    method_name = protocol_method_name(protocol)
    if ds_name == 'AIRPOLLUTION':
        if not isinstance(preprocessing_metadata, dict):
            raise ValueError('Air Pollution artifacts require preprocessing metadata.')
        preprocessing_method = preprocessing_metadata.get('method')
        if preprocessing_method not in UNLABELED_SCALING_METHODS:
            raise ValueError(
                'Air Pollution artifact has an unsupported preprocessing method: '
                f'{preprocessing_method!r}.'
            )
        expected_range = [0.0, 1.0] if preprocessing_method == 'minmax' else None
        if preprocessing_metadata.get('feature_range') != expected_range:
            raise ValueError(
                'Air Pollution preprocessing feature_range does not match method '
                f'{preprocessing_method!r}.'
            )
        if preprocessing_metadata.get('feature_columns') != feature_columns:
            raise ValueError(
                'Air Pollution preprocessing columns must match artifact '
                'feature columns.'
            )

    h_view1 = _restore_original_order(np.asarray(h_view1), orig_idx)
    h_view2 = _restore_original_order(np.asarray(h_view2), orig_idx)
    h_fused = _restore_original_order(np.asarray(h_fused), orig_idx)
    labels = _restore_original_order(np.asarray(labels), orig_idx)
    true_labels_ordered = (
        _restore_original_order(np.asarray(true_labels), orig_idx)
        if true_labels is not None
        else None
    )
    view_concat_representation = np.concatenate([h_view1, h_view2], axis=1)
    fusion_expressions = {
        FUSION_ENCODER_AVERAGE: (
            "h_fused = (view1_latent + view2_latent) / 2"
        ),
    }
    fusion_expression = fusion_expressions[protocol.fusion_contract]

    artifact = {
        'schema_version': PUBLIC_ARTIFACT_SCHEMA_VERSION,
        'algorithm_family': 'MvDEC',
        'algorithm': method_name,
        'method_name': method_name,
        'protocol_id': protocol.protocol_id,
        'run_id': run_id,
        'config_hash': config_hash,
        'protocol_contract': protocol_contract,
        'protocol_contract_sha256': protocol_sha256,
        'claim_scope': protocol.claim_scope,
        'architecture_source': protocol.architecture_source,
        'view2_architecture_id': protocol.view2_architecture_id,
        'refinement_source': protocol.refinement_source,
        'dataset': ds_name,
        'paper_basis': [
            '2025_Multi-view Deep Embedded Clustering architecture',
            '2021_Deep Embedded K-Means Clustering refinement',
        ],
        "fusion_contract": protocol.fusion_contract,
        "view_output_layout": view_output_layout(),
        "final_training_objective": training_objective,
        "h_view1": h_view1,
        "h_view2": h_view2,
        "h_fused": h_fused,
        "view_concat_representation": view_concat_representation,
        "labels": labels,
        "true_labels": true_labels_ordered,
        "row_indices": np.arange(len(labels), dtype=int),
        "source_sha256": source_sha256,
        "init": "k-means",
        "score": float(score),
        "iteration": int(iteration),
        "input_dim": int(input_shape),
        "view1_latent_dim": int(hidden_units),
        "view2_latent_dim": int(hidden_units),
        "fusion_dim": int(h_fused.shape[1]),
        "n_clusters": int(n_clusters),
        "kmeans_n_init": int(KMEANS_N_INIT),
        "kmeans_refresh_policy": protocol.kmeans_refresh_policy,
        "refinement_batching_policy": protocol.refinement_batching_policy,
        "batches_per_epoch": int(batches_per_epoch),
        "stop_reason": stop_reason,
        "refinement_epochs_completed": int(refinement_epochs_completed),
        "n_samples": int(h_fused.shape[0]),
        "feature_columns": feature_columns,
        "eigenvalue_order": EIGENVALUE_ORDER,
        "greedy_eigen_direction": greedy_eigen_direction,
        "greedy_eigen_index": int(greedy_eigen_position),
        "greedy_target_mode": greedy_target_mode,
        "preprocessing": preprocessing_metadata,
        "config": {
            "seed": random_seed,
            "n_clusters": int(n_clusters),
            "kmeans_n_init": int(KMEANS_N_INIT),
            "batch_size": int(batch_size),
            "pretrain_epochs": int(pretrain_epochs),
            "kmeans_refresh_policy": protocol.kmeans_refresh_policy,
            "refinement_batching_policy": protocol.refinement_batching_policy,
            "batches_per_epoch": int(batches_per_epoch),
            "update_interval": int(schedule.kmeans_refresh_interval),
            "max_refinement_epochs": int(max_refinement_epochs),
            "max_training_steps": int(schedule.max_training_steps),
            "stop_reason": stop_reason,
            "refinement_epochs_completed": int(refinement_epochs_completed),
            "assignment_change_tolerance": float(assignment_change_tolerance),
            "view1_latent_dim": int(hidden_units),
            "view2_latent_dim": int(hidden_units),
            "view1_filters": list(view1_filters),
            "view2_base_units": int(view2_base_units),
            "view2_architecture_id": protocol.view2_architecture_id,
            "fusion_contract": protocol.fusion_contract,
            "view1_embedding_dim": int(h_view1.shape[1]),
            "view2_embedding_dim": int(h_view2.shape[1]),
            "fusion_dim": int(h_fused.shape[1]),
            "model_view_output_dim": int(view_output_width()),
            "model_view_output_layout": view_output_layout(),
            "final_training_objective": training_objective,
            "protocol_id": protocol.protocol_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "protocol_contract": protocol_contract,
            "protocol_contract_sha256": protocol_sha256,
            "eigenvalue_order": EIGENVALUE_ORDER,
            "greedy_eigen_direction": greedy_eigen_direction,
            "greedy_eigen_index": int(greedy_eigen_position),
            "greedy_target_mode": greedy_target_mode,
            "loss_weights": {
                "reconstruction": protocol.reconstruction_weight,
                "kmeans": protocol.kmeans_weight,
                "scatter_trace_diagnostic": protocol.scatter_trace_weight,
                "greedy": protocol.greedy_weight,
            },
            "fusion": fusion_expression,
            'preprocessing': preprocessing_metadata,
        },
    }
    if external_metrics is not None:
        artifact["acc"] = float(external_metrics.acc)
        artifact["nmi"] = float(external_metrics.nmi)
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        if not isinstance(representation_external_metrics, dict) or set(
            representation_external_metrics
        ) != {"view1", "view2", "fused"}:
            raise ValueError(
                "Public reproduction artifacts require View1, View2, and fused "
                "external metrics."
            )
        if training_steps_completed is None:
            raise ValueError(
                "Public reproduction artifacts require completed training steps."
            )
        artifact["training_steps_completed"] = int(training_steps_completed)
        artifact["kmeans_n_init_policy"] = protocol.kmeans_n_init_policy
        artifact["config"].update(
            {
                "training_steps_completed": int(training_steps_completed),
                "kmeans_n_init_policy": protocol.kmeans_n_init_policy,
                "pretrain_loss_reduction": protocol.pretrain_loss_reduction,
                "pretrain_shuffle_buffer": protocol.pretrain_shuffle_buffer,
                "refinement_objective": protocol.refinement_objective,
            }
        )
        for representation_name, metrics in representation_external_metrics.items():
            artifact[f"{representation_name}_acc"] = float(metrics.acc)
            artifact[f"{representation_name}_nmi"] = float(metrics.nmi)
        if any(
            (
                not np.isclose(artifact["acc"], artifact["fused_acc"]),
                not np.isclose(artifact["nmi"], artifact["fused_nmi"]),
            )
        ):
            raise ValueError("Fused metrics must remain the canonical ACC/NMI aliases.")

    write_public_artifact(artifact, Path(artifact_path))
    print(f'MvDEC artifact was saved to {artifact_path}')


def model_view1(load_weights=True, weights_path=None):
    filters = view1_filters
    init = 'glorot_uniform'
    activation = 'relu'
    output_activation = 'linear'
    input = layers.Input(shape=(input_shape,))
    x = input
    for i in range(len(filters)):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(
        hidden_units,
        activation=output_activation,
        kernel_initializer=init,
    )(x)
    h = x

    for i in range(len(filters) - 1, -1, -1):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    y = layers.Dense(
        input_shape,
        activation=output_activation,
        kernel_initializer=init,
    )(x)

    output = layers.Concatenate()([h, y])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        path = (
            Path(weights_path)
            if weights_path is not None
            else Path(f'weight_base_view1_{ds_name}.weights.h5')
        )
        model.load_weights(path)
        print('model_view1: weights was loaded')
    return model


def model_view2(
    load_weights=True,
    weights_path=None,
    architecture_id=VIEW2_ARCHITECTURE_ID,
):
    """Build View2 according to the immutable protocol architecture identity."""

    if architecture_id not in {
        VIEW2_ARCHITECTURE_ID,
        VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
        VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
    }:
        raise ValueError(f"Unsupported View2 architecture: {architecture_id!r}.")

    init = 'glorot_uniform'
    activation = 'relu'
    output_activation = 'linear'
    input = layers.Input(shape=(input_shape,))
    b = view2_base_units

    e1 = layers.Dense(b, activation=activation, kernel_initializer=init)(input)
    e1 = layers.Dense(b, activation=activation, kernel_initializer=init)(e1)
    e2 = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(e1)
    e2 = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(e2)
    e3 = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(e2)
    e3 = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(e3)
    e4 = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(e3)
    e4 = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(e4)
    bottleneck = layers.Dense(
        16 * b,
        activation=activation,
        kernel_initializer=init,
    )(e4)

    if architecture_id == VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID:
        h = layers.Dense(
            hidden_units,
            activation=output_activation,
            kernel_initializer=init,
            name="view2_latent",
        )(bottleneck)
        decoder_input = h
    else:
        decoder_input = bottleneck

    x = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(
        decoder_input
    )
    x = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e4])
    x = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e3])
    x = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e2])
    x = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(b // 2, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e1])
    x = layers.Dense(b, activation=activation, kernel_initializer=init)(x)
    if architecture_id == VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID:
        output = layers.Dense(
            hidden_units + input_shape,
            activation=output_activation,
            kernel_initializer=init,
            name="view2_joint_head",
        )(x)
    else:
        if architecture_id == VIEW2_ARCHITECTURE_ID:
            h = layers.Dense(
                hidden_units,
                activation=output_activation,
                kernel_initializer=init,
                name="view2_latent",
            )(x)
            reconstruction_input = h
        else:
            reconstruction_input = x
        y = layers.Dense(
            input_shape,
            activation=output_activation,
            kernel_initializer=init,
            name="view2_reconstruction",
        )(reconstruction_input)
        output = layers.Concatenate(name="view2_output")([h, y])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        path = (
            Path(weights_path)
            if weights_path is not None
            else Path(f'weight_base_view2_{ds_name}.weights.h5')
        )
        model.load_weights(path)
        print('model_view2: weights was loaded')
    return model


def squared_euclidean_per_sample(y_true, y_pred):
    return tf.reduce_sum(tf.math.squared_difference(y_true, y_pred), axis=-1)


def greedy_loss_components(y_true, y_pred, eigen_index):
    squared_error = tf.math.squared_difference(y_true, y_pred)
    selected_per_sample = squared_error[:, eigen_index]
    total_per_sample = tf.reduce_sum(squared_error, axis=-1)
    selected_loss = tf.reduce_mean(selected_per_sample)
    nonselected_loss = tf.reduce_mean(total_per_sample - selected_per_sample)
    return selected_loss, nonselected_loss


def release_greedy_loss_components(y_true, y_pred, eigen_index):
    """Partition released MSE while preserving its mean-over-dimensions scale."""

    selected_loss, nonselected_loss = greedy_loss_components(
        y_true,
        y_pred,
        eigen_index,
    )
    latent_width = tf.cast(tf.shape(y_pred)[-1], y_pred.dtype)
    return selected_loss / latent_width, nonselected_loss / latent_width


def selected_direction_greedy_loss(
    transformed_embeddings,
    transformed_centroids,
    eigen_index,
):
    return tf.reduce_mean(
        tf.math.squared_difference(
            transformed_embeddings[:, eigen_index],
            transformed_centroids[:, eigen_index],
        )
    )


def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = reconstruction_output(y_pred)
    return squared_euclidean_per_sample(y_true, y_pred)


def release_loss_train_base(y_true, y_pred):
    """Match the released autoencoder MSE reduction over input dimensions."""

    y_true = layers.Flatten()(y_true)
    y_pred = reconstruction_output(y_pred)
    return tf.keras.losses.mse(y_true, y_pred)


def release_reconstruction_greedy_losses(
    x_batch,
    y_pred1,
    y_pred2,
    y_true,
    y_pred_cluster,
    reconstruction_weight,
    greedy_weight,
):
    """Return release-scaled L1, L4, and their weighted refinement objective."""

    reconstruction_loss = release_loss_train_base(
        x_batch,
        y_pred1,
    ) + release_loss_train_base(x_batch, y_pred2)
    greedy_loss = tf.keras.losses.mse(y_true, y_pred_cluster)
    total_loss = (
        reconstruction_weight * reconstruction_loss + greedy_weight * greedy_loss
    )
    return total_loss, reconstruction_loss, greedy_loss


def pretraining_loss(protocol: MvdecProtocol):
    """Return the loss function fixed by one MvDEC protocol."""

    if protocol.pretrain_loss_reduction == SUM_SQUARED_PRETRAIN_REDUCTION:
        return loss_train_base
    if protocol.pretrain_loss_reduction == RELEASE_PRETRAIN_REDUCTION:
        return release_loss_train_base
    raise ValueError(
        f"Unsupported pretraining loss reduction: {protocol.pretrain_loss_reduction!r}."
    )


class PretrainProgressCallback(tf.keras.callbacks.Callback):
    """Print compact, immediately flushed pretraining progress."""

    def __init__(self, view_name: str, total_epochs: int, interval: int) -> None:
        super().__init__()
        self.view_name = view_name
        self.total_epochs = total_epochs
        self.interval = validate_progress_interval(interval)
        self.started_at = None

    def on_train_begin(self, logs=None):
        self.started_at = time.time()

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        if not should_log_progress(completed, self.total_epochs, self.interval):
            return
        loss = float((logs or {}).get('loss', np.nan))
        elapsed = time.time() - self.started_at
        print(
            f'progress:pretrain_{self.view_name}; dataset:{ds_name}; '
            f'epoch:{completed}/{self.total_epochs}; loss:{loss:.5f}; '
            f'elapsed:{elapsed:.1f}s',
            flush=True,
        )


def train_base_view1(
    ds_xx,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
    weights_path=None,
    protocol=PRIMARY_MVDEC_PROTOCOL,
):
    model = model_view1(load_weights=False)
    model.compile(optimizer="adam", loss=pretraining_loss(protocol))
    history = model.fit(
        ds_xx,
        epochs=pretrain_epochs,
        verbose=0,
        shuffle=False,
        callbacks=[
            PretrainProgressCallback('view1', pretrain_epochs, progress_interval)
        ],
    )
    print(
        f'pretrain view1 {ds_name}: epochs={pretrain_epochs}; '
        f'final_loss={history.history["loss"][-1]:.5f}',
        flush=True,
    )
    path = (
        Path(weights_path)
        if weights_path is not None
        else Path(f'weight_base_view1_{ds_name}.weights.h5')
    )
    model.save_weights(path)


def train_base_view2(
    ds_xx,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
    weights_path=None,
    protocol=PRIMARY_MVDEC_PROTOCOL,
):
    model = model_view2(
        load_weights=False,
        architecture_id=protocol.view2_architecture_id,
    )
    model.compile(optimizer="adam", loss=pretraining_loss(protocol))
    history = model.fit(
        ds_xx,
        epochs=pretrain_epochs,
        verbose=0,
        shuffle=False,
        callbacks=[
            PretrainProgressCallback('view2', pretrain_epochs, progress_interval)
        ],
    )
    print(
        f'pretrain view2 {ds_name}: epochs={pretrain_epochs}; '
        f'final_loss={history.history["loss"][-1]:.5f}',
        flush=True,
    )
    path = (
        Path(weights_path)
        if weights_path is not None
        else Path(f'weight_base_view2_{ds_name}.weights.h5')
    )
    model.save_weights(path)


def sorted_eig(X):
    X = (X + X.T) / 2
    e_vals, e_vecs = np.linalg.eigh(X)
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def _metric_for_labels(features, labels, y=None):
    if y is not None:
        metrics = compute_external_metrics(y, labels, n_clusters=n_clusters)
        return f'acc, nmi = {metrics.acc, metrics.nmi}', (metrics.acc, metrics.nmi)
    silhouette = silhouette_score(features, labels)
    return f'silhouette = {silhouette}', silhouette


def _training_metric_for_labels(
    features,
    labels,
    y,
    protocol: MvdecProtocol,
):
    """Defer public-reproduction ground-truth metrics until final evaluation."""

    if y is not None and protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        return 'external_metrics = deferred_to_final', None
    return _metric_for_labels(features, labels, y=y)


def recompute_final_clustering(
    model1,
    model2,
    x,
    random_seed,
    fusion_contract=FUSION_ENCODER_AVERAGE,
):
    """Recompute final latents and K-Means labels for one fusion contract."""

    view1_output = model1(x).numpy()
    view2_output = model2(x).numpy()
    h1 = latent_embedding(view1_output)
    h2 = latent_embedding(view2_output)
    h_fused = fused_latent_embedding(
        view1_output,
        view2_output,
        fusion_contract,
    )
    labels = make_kmeans(random_seed).fit(h_fused).labels_
    return h1, h2, h_fused, labels


@dataclass(frozen=True)
class PublicRepresentationEvaluation:
    """Independent K-means assignment and external metrics for one latent space."""

    labels: np.ndarray
    metrics: ExternalMetrics


def evaluate_public_representations(
    h_view1: np.ndarray,
    h_view2: np.ndarray,
    h_fused: np.ndarray,
    true_labels: np.ndarray,
    random_seed: int,
) -> dict[str, PublicRepresentationEvaluation]:
    """Evaluate View1, View2, and fused embeddings without sharing assignments."""

    representations = {
        "view1": np.asarray(h_view1),
        "view2": np.asarray(h_view2),
        "fused": np.asarray(h_fused),
    }
    evaluations = {}
    for name, representation in representations.items():
        labels = make_kmeans(random_seed).fit(representation).labels_
        metrics = compute_external_metrics(
            true_labels,
            labels,
            n_clusters=n_clusters,
        )
        evaluations[name] = PublicRepresentationEvaluation(labels, metrics)
    return evaluations


def _cluster_sizes(labels):
    return np.bincount(np.asarray(labels), minlength=n_clusters).tolist()


def _loss_scalar(value):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    return float(np.mean(value))


def mean_epoch_losses(batch_losses):
    if not batch_losses:
        raise ValueError('At least one batch loss is required.')
    total_samples = sum(losses['sample_count'] for losses in batch_losses)
    return {
        name: float(
            sum(
                losses[name] * losses['sample_count'] for losses in batch_losses
            )
            / total_samples
        )
        for name in batch_losses[0]
        if name != 'sample_count'
    }


def _input_space_silhouette(input_space_reference, labels):
    if input_space_reference is None:
        return None
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(silhouette_score(input_space_reference, labels))


def _log_training_phase(
    phase,
    space,
    metric_str,
    loss,
    n_change_assignment,
    labels,
    train_start_time,
    input_space_reference=None,
    extra_fields=None,
    file_name=None,
    log_path=None,
    print_console=True,
):
    extra_str = ''
    if extra_fields:
        extra_str = ''.join(f'; {key}:{value}' for key, value in extra_fields.items())
    input_space_str = ''
    input_space_silhouette = _input_space_silhouette(input_space_reference, labels)
    if input_space_silhouette is not None:
        input_space_str = f'; silhouette_input_space = {input_space_silhouette}'
    log_str = (
        f'phase:{phase}; space:{space}; {metric_str}{input_space_str}; '
        f'loss:{loss}; '
        f'n_changed_assignment:{n_change_assignment}; '
        f'cluster_sizes:{_cluster_sizes(labels)}{extra_str}; '
        f'time:{time.time() - train_start_time:.3f}'
    )
    if print_console:
        print(log_str, flush=True)
    fields = log_str.split(';')
    if log_path is not None:
        append_run_log(Path(log_path), fields)
    else:
        log_csv(fields, file_name=ds_name if file_name is None else file_name)


def train(
    x,
    y=None,
    raw_x=None,
    orig_idx=None,
    feature_columns=None,
    preprocessing_metadata=None,
    greedy_eigen_direction=DEFAULT_GREEDY_EIGEN_DIRECTION,
    greedy_target_mode=DEFAULT_GREEDY_TARGET_MODE,
    max_refinement_epochs=MAX_REFINEMENT_EPOCHS,
    random_seed=None,
    time_start=None,
    artifact_path=None,
    source_sha256=None,
    progress_interval=DEFAULT_PROGRESS_INTERVAL,
    protocol=PRIMARY_MVDEC_PROTOCOL,
    run_id=None,
    config_hash=None,
    training_log_path=None,
    pretrain_view1_path=None,
    pretrain_view2_path=None,
    final_view1_path=None,
    final_view2_path=None,
    assignments_path=None,
    return_run_metrics=False,
):
    train_start_time = time.time() if time_start is None else time_start
    if any(
        (
            protocol.greedy_eigen_direction != greedy_eigen_direction,
            protocol.greedy_target_mode != greedy_target_mode,
            protocol.kmeans_weight != lambda_kmeans,
            protocol.greedy_weight != lambda_greedy,
        )
    ):
        raise ValueError('MvDEC training settings do not match its protocol contract.')
    greedy_index = greedy_eigen_index(greedy_eigen_direction)
    validate_greedy_target_mode(greedy_target_mode)
    max_refinement_epochs = validate_protocol_max_refinement_epochs(
        protocol,
        max_refinement_epochs,
    )
    progress_interval = validate_progress_interval(progress_interval)
    schedule = resolve_refinement_schedule(
        protocol,
        len(x),
        batch_size,
        max_refinement_epochs,
    )
    batches_per_epoch = schedule.batches_per_epoch
    kmeans_refresh_interval = schedule.kmeans_refresh_interval
    max_training_steps = schedule.max_training_steps
    batch_rng = np.random.default_rng(random_seed)
    experiment_tag = (
        f'{protocol.protocol_id}_{greedy_eigen_direction}_eigen_{greedy_target_mode}'
        f'{loss_ablation_tag(lambda_kmeans, lambda_greedy)}'
    )
    train_log_name = run_id or f'{ds_name}_{experiment_tag.upper()}'
    log_str = (
        'phase; space; metric; silhouette_input_space; loss; '
        'n_changed_assignment; cluster_sizes; '
        'greedy_eigen_direction; greedy_eigen_index; greedy_eigenvalue; '
        'greedy_target_mode; '
        f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    )
    if training_log_path is not None:
        append_run_log(Path(training_log_path), log_str.split(';'))
    else:
        log_csv(log_str.split(';'), file_name=train_log_name)
    model1 = model_view1(weights_path=pretrain_view1_path)
    model2 = model_view2(
        weights_path=pretrain_view2_path,
        architecture_id=protocol.view2_architecture_id,
    )

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    reconstruction_loss_value = 0
    kmeans_loss_value = 0
    orthonormal_loss_value = 0
    greedy_loss_value = 0
    greedy_selected_loss_value = 0
    greedy_nonselected_loss_value = 0
    silhouette = np.nan
    acc = np.nan
    nmi = np.nan
    index = 0
    kmeans_n_init = KMEANS_N_INIT
    stop_reason = "max_epochs_reached"
    refinement_epochs_completed = 0
    training_steps_completed = 0
    epoch_batch_losses = []
    last_epoch_loss_means = None
    assignment = np.array([-1] * len(x))
    epoch_batches = None
    if raw_x is not None and raw_x.shape != x.shape:
        raise ValueError('raw_x and x must have the same shape.')

    if raw_x is not None:
        raw_kmeans = make_kmeans(random_seed).fit(raw_x)
        raw_metric_str, _ = _training_metric_for_labels(
            raw_x,
            raw_kmeans.labels_,
            y,
            protocol,
        )
        _log_training_phase(
            phase='baseline_raw_input',
            space='x_raw',
            metric_str=raw_metric_str,
            loss=0.0,
            n_change_assignment=len(x),
            labels=raw_kmeans.labels_,
            train_start_time=train_start_time,
            input_space_reference=x if y is None else None,
            file_name=train_log_name,
            log_path=training_log_path,
        )

    input_kmeans = make_kmeans(random_seed).fit(x)
    input_metric_str, _ = _training_metric_for_labels(
        x,
        input_kmeans.labels_,
        y,
        protocol,
    )
    input_phase = 'baseline_scaled_input' if raw_x is not None else 'baseline_raw_input'
    input_space = _preprocessing_input_space(preprocessing_metadata)
    _log_training_phase(
        phase=input_phase,
        space=input_space,
        metric_str=input_metric_str,
        loss=0.0,
        n_change_assignment=len(x),
        labels=input_kmeans.labels_,
        train_start_time=train_start_time,
        input_space_reference=x if y is None else None,
        file_name=train_log_name,
        log_path=training_log_path,
    )
    for ite in range(max_training_steps):
        log_paper_step_checkpoint = ite % kmeans_refresh_interval == 0
        log_step9_end_of_epoch = is_refinement_epoch_end(
            ite,
            kmeans_refresh_interval,
        )
        if (
            protocol.refinement_batching_policy == REFINEMENT_BATCHING_POLICY
            and ite % batches_per_epoch == 0
        ):
            epoch_batches = epoch_batch_indices(len(x), batch_size, batch_rng)
        if log_paper_step_checkpoint:
            epoch_batch_losses = []
            view1_output = model1(x).numpy()
            view2_output = model2(x).numpy()
            H = fused_latent_embedding(
                view1_output,
                view2_output,
                protocol.fusion_contract,
            )
            checkpoint_kmeans_n_init = kmeans_n_init
            ans_kmeans = make_kmeans(
                random_seed,
                n_init=checkpoint_kmeans_n_init,
            ).fit(H)
            kmeans_n_init = next_kmeans_n_init(protocol, ans_kmeans)

            U = ans_kmeans.cluster_centers_
            assignment_new = ans_kmeans.labels_
            n_change_assignment = count_aligned_assignment_changes(
                assignment,
                assignment_new,
            )
            assignment = assignment_new

            S_i = []
            for i in range(n_clusters):
                temp = H[assignment == i] - U[i]
                temp = np.matmul(np.transpose(temp), temp)
                S_i.append(temp)
            S_i = np.array(S_i)
            S = np.sum(S_i, 0)
            Evals, V = sorted_eig(S)
            H_vt = np.matmul(H, V)
            U_vt = np.matmul(U, V)
            selected_eigenvalue = float(Evals[greedy_index])
            selected_eigen_position = greedy_index % len(Evals)
            H_vt_greedy_target = build_greedy_target(
                H_vt,
                U_vt,
                assignment,
                greedy_index,
            )
            eigen_log_fields = {
                "greedy_eigen_direction": greedy_eigen_direction,
                "greedy_eigen_index": selected_eigen_position,
                "greedy_eigenvalue": selected_eigenvalue,
                "greedy_target_mode": greedy_target_mode,
                "kmeans_n_init": checkpoint_kmeans_n_init,
                "kmeans_refresh_policy": protocol.kmeans_refresh_policy,
                "refinement_batching_policy": (protocol.refinement_batching_policy),
                "batches_per_epoch": batches_per_epoch,
                "lambda_kmeans": lambda_kmeans,
                "lambda_greedy": lambda_greedy,
            }
            if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
                eigen_log_fields["next_kmeans_n_init"] = kmeans_n_init
            loss = np.round(_loss_scalar(loss_value), 5)
            metric_str, metric_value = _training_metric_for_labels(
                H,
                assignment,
                y,
                protocol,
            )
            if y is not None and metric_value is not None:
                acc, nmi = metric_value
            else:
                silhouette = metric_value
            checkpoint = ite // kmeans_refresh_interval
            checkpoint_unit = (
                "epoch"
                if protocol.kmeans_refresh_policy == KMEANS_REFRESH_POLICY
                else "checkpoint"
            )
            paper_step_suffix = (
                "pre_refine" if ite == 0 else f"refine_{checkpoint_unit}_{checkpoint}"
            )
            _log_training_phase(
                phase=f'paper_step4_to_7_common_{paper_step_suffix}',
                space='H_fused_latent',
                metric_str=metric_str,
                loss=loss,
                n_change_assignment=n_change_assignment,
                labels=assignment,
                train_start_time=train_start_time,
                input_space_reference=x if y is None else None,
                extra_fields=eigen_log_fields,
                file_name=train_log_name,
                log_path=training_log_path,
                print_console=False,
            )
            if checkpoint == 0 or checkpoint % progress_interval == 0:
                print(
                    f"progress:refinement; dataset:{ds_name}; "
                    f"{checkpoint_unit}:{checkpoint}/"
                    f"{max_training_steps // kmeans_refresh_interval}; "
                    f"{metric_str}; changed:{n_change_assignment}/{len(x)}; "
                    f"elapsed:{time.time() - train_start_time:.1f}s",
                    flush=True,
                )
            greedy_metric_str, _ = _training_metric_for_labels(
                H_vt_greedy_target,
                assignment,
                y,
                protocol,
            )
            _log_training_phase(
                phase=f'paper_step8_greedy_target_Yprime_{paper_step_suffix}',
                space='Y_greedy_target',
                metric_str=greedy_metric_str,
                loss=loss,
                n_change_assignment=n_change_assignment,
                labels=assignment,
                train_start_time=train_start_time,
                extra_fields=eigen_log_fields,
                file_name=train_log_name,
                log_path=training_log_path,
                print_console=False,
            )

        if n_change_assignment <= len(x) * assignment_change_tolerance:
            stop_reason = 'converged_assignment'
            refinement_epochs_completed = (
                ite // batches_per_epoch
                if protocol.protocol_id in PUBLIC_PROTOCOL_IDS
                else ite // kmeans_refresh_interval
            )
            training_steps_completed = ite
            break
        if protocol.refinement_batching_policy == REFINEMENT_BATCHING_POLICY:
            idx = epoch_batches[index]
        else:
            idx = sequential_release_batch_indices(len(x), batch_size, ite)
        temp = assignment[idx]
        x_batch = tf.convert_to_tensor(x[idx], dtype=tf.float32)
        y_true_tensor = None
        if greedy_target_mode == 'frozen_snapshot':
            y_true = build_greedy_target(H_vt[idx], U_vt, temp, greedy_index)
            y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.float32)
        V_tensor = tf.convert_to_tensor(V, dtype=tf.float32)
        U_batch_tensor = tf.convert_to_tensor(U[temp], dtype=tf.float32)
        U_vt_batch_tensor = tf.convert_to_tensor(U_vt[temp], dtype=tf.float32)

        with tf.GradientTape() as tape:
            y_pred1 = model1(x_batch)
            y_pred2 = model2(x_batch)
            h_pred = fused_latent_embedding(
                y_pred1,
                y_pred2,
                protocol.fusion_contract,
            )
            y_pred_cluster = tf.matmul(h_pred, V_tensor)
            if protocol.refinement_objective == JOINT_REFINEMENT_OBJECTIVE:
                orthonormal_residual = tf.matmul(
                    h_pred - U_batch_tensor,
                    V_tensor,
                )
                reconstruction_loss_value = tf.reduce_mean(
                    squared_euclidean_per_sample(
                        x_batch,
                        reconstruction_output(y_pred1),
                    )
                    + squared_euclidean_per_sample(
                        x_batch,
                        reconstruction_output(y_pred2),
                    )
                )
                kmeans_loss_value = tf.reduce_mean(
                    squared_euclidean_per_sample(U_batch_tensor, h_pred)
                )
                orthonormal_loss_value = tf.reduce_mean(
                    squared_euclidean_per_sample(
                        tf.zeros_like(orthonormal_residual),
                        orthonormal_residual,
                    )
                )
                if greedy_target_mode == "selected_dimension_only":
                    greedy_selected_loss_value = selected_direction_greedy_loss(
                        y_pred_cluster,
                        U_vt_batch_tensor,
                        greedy_index,
                    )
                    greedy_nonselected_loss_value = tf.zeros_like(
                        greedy_selected_loss_value
                    )
                else:
                    (
                        greedy_selected_loss_value,
                        greedy_nonselected_loss_value,
                    ) = greedy_loss_components(
                        y_true_tensor,
                        y_pred_cluster,
                        greedy_index,
                    )
                greedy_loss_value = (
                    greedy_selected_loss_value + greedy_nonselected_loss_value
                )
                loss_value = (
                    lambda_reconstruction * reconstruction_loss_value
                    + lambda_kmeans * kmeans_loss_value
                    + lambda_orthonormal * orthonormal_loss_value
                    + lambda_greedy * greedy_loss_value
                )
            elif protocol.refinement_objective == RELEASE_REFINEMENT_OBJECTIVE:
                kmeans_loss_value = tf.constant(0.0)
                orthonormal_loss_value = tf.constant(0.0)
                (
                    greedy_selected_loss_value,
                    greedy_nonselected_loss_value,
                ) = release_greedy_loss_components(
                    y_true_tensor,
                    y_pred_cluster,
                    greedy_index,
                )
                (
                    loss_value,
                    reconstruction_loss_value,
                    greedy_loss_value,
                ) = release_reconstruction_greedy_losses(
                    x_batch,
                    y_pred1,
                    y_pred2,
                    y_true_tensor,
                    y_pred_cluster,
                    lambda_reconstruction,
                    lambda_greedy,
                )
            else:
                raise ValueError(
                    "Unsupported refinement objective: "
                    f"{protocol.refinement_objective!r}."
                )
        trainable_variables = model1.trainable_variables + model2.trainable_variables
        grads = tape.gradient(loss_value, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables, strict=False))
        epoch_batch_losses.append(
            {
                'sample_count': len(idx),
                'total': _loss_scalar(loss_value),
                'reconstruction': _loss_scalar(reconstruction_loss_value),
                'kmeans': _loss_scalar(kmeans_loss_value),
                'orthonormal': _loss_scalar(orthonormal_loss_value),
                'greedy': _loss_scalar(greedy_loss_value),
                'greedy_selected': _loss_scalar(greedy_selected_loss_value),
                'greedy_nonselected': _loss_scalar(
                    greedy_nonselected_loss_value
                ),
            }
        )
        training_steps_completed = ite + 1
        if log_step9_end_of_epoch:
            refinement_epoch = (ite + 1) // kmeans_refresh_interval
            last_epoch_loss_means = mean_epoch_losses(epoch_batch_losses)
            greedy_total_scalar = last_epoch_loss_means['greedy']
            greedy_selected_scalar = last_epoch_loss_means['greedy_selected']
            greedy_nonselected_scalar = last_epoch_loss_means[
                'greedy_nonselected'
            ]
            greedy_denominator = max(greedy_total_scalar, 1e-12)
            view1_output_after_update = model1(x).numpy()
            view2_output_after_update = model2(x).numpy()
            H_after_update = fused_latent_embedding(
                view1_output_after_update,
                view2_output_after_update,
                protocol.fusion_contract,
            )
            after_update_metric_str, _ = _training_metric_for_labels(
                H_after_update,
                assignment,
                y,
                protocol,
            )
            if protocol.kmeans_refresh_policy == KMEANS_REFRESH_POLICY:
                phase_name = f"paper_step9_after_full_epoch_{refinement_epoch}"
                metric_scope = "full_dataset_after_epoch"
                loss_scope = "epoch_mean"
            else:
                phase_name = (
                    f"paper_step9_after_release_update_window_{refinement_epoch}"
                )
                metric_scope = "full_dataset_after_release_update_window"
                loss_scope = "release_update_window_mean"
            _log_training_phase(
                phase=phase_name,
                space="H_fused_latent_same_labels",
                metric_str=after_update_metric_str,
                loss=(
                    f'total:{last_epoch_loss_means["total"]:.5f}, '
                    f'L1_reconstruction:'
                    f'{last_epoch_loss_means["reconstruction"]:.5f}, '
                    f'L2_kmeans:{last_epoch_loss_means["kmeans"]:.5f}, '
                    f'L3_trace_logged:'
                    f'{last_epoch_loss_means["orthonormal"]:.5f}, '
                    f'L4_greedy:{greedy_total_scalar:.5f}, '
                    f'L4_selected_direction:'
                    f'{greedy_selected_scalar:.5f}, '
                    f'L4_nonselected_snapshot_anchor:'
                    f'{greedy_nonselected_scalar:.5f}, '
                    f'L4_selected_fraction:'
                    f'{greedy_selected_scalar / greedy_denominator:.5f}, '
                    f'L4_anchor_fraction:'
                    f'{greedy_nonselected_scalar / greedy_denominator:.5f}'
                ),
                n_change_assignment=n_change_assignment,
                labels=assignment,
                train_start_time=train_start_time,
                extra_fields={
                    **eigen_log_fields,
                    "refinement_epoch": refinement_epoch,
                    "step9_metric_scope": metric_scope,
                    "step9_loss_scope": loss_scope,
                    "step9_loss_batches": len(epoch_batch_losses),
                    "step9_loss_samples": sum(
                        losses["sample_count"] for losses in epoch_batch_losses
                    ),
                },
                file_name=train_log_name,
                log_path=training_log_path,
                print_console=False,
            )

        if protocol.refinement_batching_policy == REFINEMENT_BATCHING_POLICY:
            index = (index + 1) % batches_per_epoch

    if stop_reason == "max_epochs_reached":
        training_steps_completed = max_training_steps
        refinement_epochs_completed = training_steps_completed // batches_per_epoch
    resolved_final_view1_path = (
        Path(final_view1_path)
        if final_view1_path is not None
        else Path(f'weight_final_view1_{ds_name}_{experiment_tag}.weights.h5')
    )
    resolved_final_view2_path = (
        Path(final_view2_path)
        if final_view2_path is not None
        else Path(f'weight_final_view2_{ds_name}_{experiment_tag}.weights.h5')
    )
    model1.save_weights(resolved_final_view1_path)
    model2.save_weights(resolved_final_view2_path)
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        stop_log_str = (
            f"phase:training_stop; stop_reason:{stop_reason}; "
            f"training_steps_completed:{training_steps_completed}; "
            f"max_training_steps:{max_training_steps}; "
            f"time:{time.time() - train_start_time:.3f}"
        )
    else:
        stop_log_str = (
            f"phase:training_stop; stop_reason:{stop_reason}; "
            f"refinement_epochs_completed:{refinement_epochs_completed}; "
            f"max_refinement_epochs:{max_refinement_epochs}; "
            f"time:{time.time() - train_start_time:.3f}"
        )
    print(stop_log_str)
    if training_log_path is not None:
        append_run_log(Path(training_log_path), stop_log_str.split(';'))
    else:
        log_csv(stop_log_str.split(';'), file_name=train_log_name)

    view1_output = model1(x).numpy()
    view2_output = model2(x).numpy()
    h1 = latent_embedding(view1_output)
    h2 = latent_embedding(view2_output)
    H = fused_latent_embedding(
        view1_output,
        view2_output,
        protocol.fusion_contract,
    )
    representation_evaluations = None
    if y is not None and protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        representation_evaluations = evaluate_public_representations(
            h1,
            h2,
            H,
            y,
            random_seed,
        )
        final_assignment = representation_evaluations["fused"].labels
    else:
        final_assignment = make_kmeans(random_seed).fit(H).labels_
    final_n_change_assignment = count_aligned_assignment_changes(
        assignment,
        final_assignment,
    )
    assignment = final_assignment
    if representation_evaluations is not None:
        fused_metrics = representation_evaluations["fused"].metrics
        metric_str = f"acc, nmi = {fused_metrics.acc, fused_metrics.nmi}"
        final_metric = (fused_metrics.acc, fused_metrics.nmi)
    else:
        metric_str, final_metric = _metric_for_labels(H, assignment, y=y)
    artifact_score = float(silhouette_score(H, assignment))
    external_metrics = None
    if y is None:
        silhouette = final_metric
    else:
        acc, nmi = final_metric
        external_metrics = (
            representation_evaluations["fused"].metrics
            if representation_evaluations is not None
            else compute_external_metrics(
                y,
                assignment,
                n_clusters=n_clusters,
            )
        )
    final_phase = (
        'final_recomputed_artifact'
        if y is None and orig_idx is not None
        else 'final_recomputed_state'
    )
    final_log_fields = {
        "protocol_id": protocol.protocol_id,
        "greedy_eigen_direction": greedy_eigen_direction,
        "greedy_eigen_index": (
            H.shape[1] - 1 if greedy_eigen_direction == "largest" else 0
        ),
        "greedy_target_mode": greedy_target_mode,
        "lambda_kmeans": lambda_kmeans,
        "lambda_greedy": lambda_greedy,
        "kmeans_refresh_policy": protocol.kmeans_refresh_policy,
        "batches_per_epoch": batches_per_epoch,
        "stop_reason": stop_reason,
        "refinement_epochs_completed": refinement_epochs_completed,
    }
    if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
        final_log_fields.update(
            {
                "training_steps_completed": training_steps_completed,
                "max_training_steps": max_training_steps,
                "loss_scope": (
                    "last_completed_update_window_mean"
                    if protocol.protocol_id == PUBLIC_DEKM_CONSISTENT_PROTOCOL_ID
                    else "last_completed_release_update_window_mean"
                ),
            }
        )
    else:
        final_log_fields.update(
            {
                "max_refinement_epochs": max_refinement_epochs,
                "loss_scope": "last_completed_epoch_mean",
            }
        )
    _log_training_phase(
        phase=final_phase,
        space='H_fused_latent',
        metric_str=metric_str,
        loss=np.round(last_epoch_loss_means['total'], 5),
        n_change_assignment=final_n_change_assignment,
        labels=assignment,
        train_start_time=train_start_time,
        input_space_reference=x if y is None else None,
        extra_fields=final_log_fields,
        file_name=train_log_name,
        log_path=training_log_path,
    )

    if orig_idx is not None:
        if artifact_path is None:
            raise ValueError('MvDEC train requires a run-owned artifact path.')
        h_ordered = _restore_original_order(H, orig_idx)
        labels_ordered = _restore_original_order(assignment, orig_idx)
        artifact_path = Path(artifact_path)
        if y is None:
            result = pd.DataFrame(
                h_ordered,
                columns=[f'h_{i}' for i in range(H.shape[1])],
            )
            result.insert(0, 'orig_index', np.arange(len(orig_idx)))
            result.insert(1, 'protocol_id', protocol.protocol_id)
            result.insert(2, 'run_id', run_id)
            result.insert(3, 'config_hash', config_hash)
            result['cluster'] = labels_ordered
            assignments_path = (
                Path(assignments_path)
                if assignments_path is not None
                else Path('output') / f'{ds_name}_{experiment_tag}_clusters.csv'
            )
        else:
            result = public_assignment_frame(
                dataset=ds_name,
                method=protocol_method_name(protocol),
                labels=labels_ordered,
                true_labels=_restore_original_order(y, orig_idx),
                random_seed=random_seed,
            )
            result.insert(2, 'protocol_id', protocol.protocol_id)
            result.insert(3, 'run_id', run_id)
            result.insert(4, 'config_hash', config_hash)
            assignments_path = (
                Path(assignments_path)
                if assignments_path is not None
                else artifact_path.with_name(f'{artifact_path.stem}_assignments.csv')
            )
        write_public_assignments(
            result,
            assignments_path,
        )
        save_airpollution_mvdec_artifact(
            artifact_path=artifact_path,
            h_view1=h1,
            h_view2=h2,
            h_fused=H,
            labels=assignment,
            score=artifact_score,
            iteration=(
                training_steps_completed
                if protocol.protocol_id in PUBLIC_PROTOCOL_IDS
                else refinement_epochs_completed
            ),
            orig_idx=orig_idx,
            feature_columns=feature_columns,
            random_seed=random_seed,
            greedy_eigen_direction=greedy_eigen_direction,
            greedy_target_mode=greedy_target_mode,
            batches_per_epoch=batches_per_epoch,
            max_refinement_epochs=max_refinement_epochs,
            stop_reason=stop_reason,
            refinement_epochs_completed=refinement_epochs_completed,
            preprocessing_metadata=preprocessing_metadata,
            true_labels=y,
            source_sha256=source_sha256,
            external_metrics=external_metrics,
            representation_external_metrics=(
                {
                    name: evaluation.metrics
                    for name, evaluation in representation_evaluations.items()
                }
                if representation_evaluations is not None
                else None
            ),
            training_steps_completed=training_steps_completed,
            protocol=protocol,
            run_id=run_id,
            config_hash=config_hash,
        )

    if y is not None:
        run_metrics = {"acc": float(acc), "nmi": float(nmi)}
        if representation_evaluations is not None:
            for name, evaluation in representation_evaluations.items():
                run_metrics[f"{name}_acc"] = float(evaluation.metrics.acc)
                run_metrics[f"{name}_nmi"] = float(evaluation.metrics.nmi)
        return run_metrics if return_run_metrics else (acc, nmi)
    run_metrics = {"silhouette": float(silhouette)}
    return run_metrics if return_run_metrics else silhouette


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='select dataset:AIRPOLLUTION,TIKI,REUTERS,20NEWS,RCV1',
    )
    parser.add_argument('ds_name', default='AIRPOLLUTION')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='First explicit run seed; subsequent runs use seed + run index.',
    )
    parser.add_argument(
        '--protocol',
        choices=PROTOCOL_IDS,
        default=PRIMARY_PROTOCOL_ID,
        help=(
            "Immutable DEKM-consistent protocol, separate public reproduction "
            "contract, or custom for an explicit ablation."
        ),
    )
    parser.add_argument(
        '--artifact-path',
        default=None,
        help=(
            'Legacy private-output anchor. When set, its parent/stem becomes '
            'the run root; artifacts still use isolated seed directories.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output/mvdec_runs'),
        help='Root directory for isolated private-dataset MvDEC runs.',
    )
    parser.add_argument('--dataset-root', type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument(
        '--public-output-dir',
        type=Path,
        default=PUBLIC_BENCHMARK_OUTPUT_DIR / 'mvdec',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite files owned by the exact same dataset/config/seed run_id.',
    )
    parser.add_argument(
        '--max-refinement-epochs',
        type=int,
        default=MAX_REFINEMENT_EPOCHS,
        help=(
            "Safety cap for primary/custom full-epoch refinement. The public "
            "reproduction protocol fixes 14,000 updates."
        ),
    )
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help="Print one heartbeat every N epochs or release checkpoints.",
    )
    parser.add_argument(
        '--preprocessing',
        choices=UNLABELED_SCALING_METHODS,
        default=None,
        help=(
            'Override preprocessing for AIRPOLLUTION or TIKI. Defaults to the '
            'dataset contract (AIRPOLLUTION=minmax, TIKI=none).'
        ),
    )
    parser.add_argument(
        '--assignment-change-tolerance',
        type=float,
        default=None,
        help=(
            'Assignment-change stopping fraction. Defaults to 0.01 for '
            'unlabeled case studies and 0.001 for public DEKM benchmarks.'
        ),
    )
    parser.add_argument(
        '--greedy-eigen-direction',
        choices=GREEDY_EIGEN_DIRECTIONS,
        default=DEFAULT_GREEDY_EIGEN_DIRECTION,
        help=(
            'Largest is the DEKM-consistent last/least-informative direction; '
            'use --protocol custom for smallest-eigen sensitivity runs.'
        ),
    )
    parser.add_argument(
        '--greedy-target-mode',
        choices=GREEDY_TARGET_MODES,
        default=DEFAULT_GREEDY_TARGET_MODE,
        help=(
            'Frozen snapshot matches the released DEKM implementation; use '
            '--protocol custom for selected-dimension-only ablations.'
        ),
    )
    parser.add_argument(
        '--lambda-kmeans',
        type=float,
        default=DEFAULT_LAMBDA_KMEANS,
        help=(
            'L2 K-means weight. The primary protocol fixes it at 0 because '
            'DEKM 2021 selects greedy last-direction refinement and L3 is '
            'trace-equivalent to L2. Use --protocol custom for ablations.'
        ),
    )
    parser.add_argument(
        '--lambda-greedy',
        type=float,
        default=DEFAULT_LAMBDA_GREEDY,
        help=(
            'L4 greedy weight, fixed at 1 by the primary protocol. Use '
            '--protocol custom to ablate it.'
        ),
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    protocol = resolve_mvdec_protocol(
        args.protocol,
        args.lambda_kmeans,
        args.lambda_greedy,
        args.greedy_eigen_direction,
        args.greedy_target_mode,
    )
    lambda_kmeans = protocol.kmeans_weight
    lambda_greedy = protocol.greedy_weight
    validate_protocol_max_refinement_epochs(
        protocol,
        args.max_refinement_epochs,
    )
    validate_progress_interval(args.progress_interval)
    if args.ds_name is None or args.ds_name not in [
        'AIRPOLLUTION',
        'TIKI',
        'REUTERS',
        '20NEWS',
        'RCV1',
    ]:
        ds_name = 'AIRPOLLUTION'
    else:
        ds_name = args.ds_name
    validate_protocol_dataset_scope(protocol, ds_name)

    if ds_name in UNLABELED_DATASETS:
        input_shape = UNLABELED_DATASETS[ds_name]['input_shape']
        n_clusters = UNLABELED_DATASETS[ds_name]['n_clusters']
        hidden_units = UNLABELED_DATASETS[ds_name]['hidden_units']
        view1_filters = UNLABELED_DATASETS[ds_name]['view1_filters']
        view2_base_units = UNLABELED_DATASETS[ds_name]['view2_base_units']
    elif ds_name == 'REUTERS':
        input_shape = 2000
        n_clusters = 4
        hidden_units = 10
    elif ds_name == '20NEWS':
        input_shape = 2000
        n_clusters = 20
        hidden_units = 10
    elif ds_name == 'RCV1':
        input_shape = 2000
        n_clusters = 4
        hidden_units = 10

    pretrain_epochs = 200
    pretrain_batch_size = 256
    batch_size = 256
    assignment_change_tolerance = resolve_assignment_change_tolerance(
        ds_name,
        args.assignment_change_tolerance,
    )
    validate_protocol_assignment_change_tolerance(
        protocol,
        ds_name,
        assignment_change_tolerance,
    )
    private_output_root = args.output_dir
    if args.artifact_path is not None:
        legacy_anchor = Path(args.artifact_path)
        private_output_root = legacy_anchor.parent / legacy_anchor.stem

    public_dataset = (
        None
        if ds_name in UNLABELED_DATASETS
        else load_dekm_public_dataset(ds_name, args.dataset_root)
    )
    run_metrics = []
    config_dirs = set()
    output_root = (
        private_output_root
        if ds_name in UNLABELED_DATASETS
        else args.public_output_dir
    )
    time_all_start = time.time()
    for run_index in range(args.runs):
        run_seed = None if args.seed is None else args.seed + run_index
        runtime_metadata = configure_deterministic_runtime(run_seed)
        time_start = time.time()
        print(
            f'start:dataset:{ds_name}; run:{run_index + 1}/{args.runs}; '
            f'seed:{run_seed}',
            flush=True,
        )
        orig_idx = None
        feature_columns = None
        raw_x = None
        preprocessing_metadata = None
        source_sha256 = None
        if ds_name in UNLABELED_DATASETS:
            dataset_config = UNLABELED_DATASETS[ds_name]
            scaling_method = resolve_unlabeled_scaling_method(
                ds_name,
                args.preprocessing,
            )
            x, orig_idx, feature_columns, source_x, preprocessing_metadata = (
                get_x_unlabeled_csv(
                    dataset_config['csv_path'],
                    shuffle_seed=run_seed,
                    scaling_method=scaling_method,
                    include_preprocessing=True,
                )
            )
            if preprocessing_metadata['method'] != 'none':
                raw_x = source_x
            source_sha256 = preprocessing_metadata['source_sha256']
            y = None
        else:
            row_indices = tf.random.shuffle(
                np.arange(len(public_dataset.X)), seed=run_seed
            ).numpy()
            x = public_dataset.X[row_indices]
            y = public_dataset.y[row_indices]
            orig_idx = row_indices
            feature_columns = [
                f'feature_{index + 1}' for index in range(public_dataset.X.shape[1])
            ]
            source_sha256 = public_dataset.source_sha256
        run_config = resolved_run_config(
            protocol=protocol,
            dataset_name=ds_name,
            tolerance=assignment_change_tolerance,
            max_refinement_epochs=args.max_refinement_epochs,
            source_sha256=source_sha256,
            feature_columns=feature_columns,
            preprocessing_metadata=preprocessing_metadata,
        )
        run_paths = resolve_run_paths(
            output_root=output_root,
            dataset=ds_name,
            protocol_id=protocol.protocol_id,
            config=run_config,
            seed=run_seed,
            run_index=run_index + 1,
        )
        prepare_run_directory(run_paths, force=args.force)
        config_dirs.add(run_paths.config_dir)
        manifest_base = {
            'run_id': run_paths.run_id,
            'status': 'running',
            'dataset': ds_name,
            'method': protocol_method_name(protocol),
            'protocol_id': protocol.protocol_id,
            'config_hash': run_paths.config_hash,
            'seed': run_seed,
            'run_index': run_index + 1,
            'started_at_utc': datetime.now(UTC).isoformat(),
            'runtime': runtime_metadata,
            'config': run_config,
            'paths': {
                'artifact': run_paths.artifact.name,
                'assignments': run_paths.assignments.name,
                'pretrain_view1': run_paths.pretrain_view1.name,
                'pretrain_view2': run_paths.pretrain_view2.name,
                'final_view1': run_paths.final_view1.name,
                'final_view2': run_paths.final_view2.name,
                'training_log': run_paths.training_log.name,
            },
        }
        write_run_manifest(run_paths.manifest, manifest_base)
        ds_xx = make_pretraining_dataset(x, run_seed, protocol=protocol)
        try:
            train_base_view1(
                ds_xx,
                args.progress_interval,
                weights_path=run_paths.pretrain_view1,
                protocol=protocol,
            )
            train_base_view2(
                ds_xx,
                args.progress_interval,
                weights_path=run_paths.pretrain_view2,
                protocol=protocol,
            )
            metric = train(
                x,
                y=y,
                raw_x=raw_x,
                orig_idx=orig_idx,
                feature_columns=feature_columns,
                preprocessing_metadata=preprocessing_metadata,
                greedy_eigen_direction=args.greedy_eigen_direction,
                greedy_target_mode=args.greedy_target_mode,
                max_refinement_epochs=args.max_refinement_epochs,
                random_seed=run_seed,
                time_start=time_start,
                artifact_path=run_paths.artifact,
                source_sha256=source_sha256,
                progress_interval=args.progress_interval,
                protocol=protocol,
                run_id=run_paths.run_id,
                config_hash=run_paths.config_hash,
                training_log_path=run_paths.training_log,
                pretrain_view1_path=run_paths.pretrain_view1,
                pretrain_view2_path=run_paths.pretrain_view2,
                final_view1_path=run_paths.final_view1,
                final_view2_path=run_paths.final_view2,
                assignments_path=run_paths.assignments,
                return_run_metrics=True,
            )
        except Exception as error:
            write_run_manifest(
                run_paths.manifest,
                {
                    **manifest_base,
                    'status': 'failed',
                    'runtime_seconds': time.time() - time_start,
                    'finished_at_utc': datetime.now(UTC).isoformat(),
                    'error': {
                        'type': type(error).__name__,
                        'message': str(error),
                    },
                },
            )
            raise
        run_metrics.append(metric)
        if y is None:
            metrics_payload = metric
            run_str = (
                f"run {run_index + 1}/{args.runs}; seed:{run_seed}; "
                f"run_id:{run_paths.run_id}; "
                f"protocol_id:{protocol.protocol_id}; "
                f"greedy_eigen_direction:{args.greedy_eigen_direction}; "
                f"greedy_target_mode:{args.greedy_target_mode}; "
                f"silhouette:{metric['silhouette']}; "
                f"time:{time.time() - time_start:.3f}"
            )
        else:
            metrics_payload = metric
            acc = metric["acc"]
            nmi = metric["nmi"]
            run_str = (
                f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
                f'run_id:{run_paths.run_id}; '
                f'protocol_id:{protocol.protocol_id}; '
                f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
                f'greedy_target_mode:{args.greedy_target_mode}; '
                f'acc:{acc}; nmi:{nmi}; time:{time.time() - time_start:.3f}'
            )
            if protocol.protocol_id in PUBLIC_PROTOCOL_IDS:
                run_str += (
                    f"; view1_acc:{metric['view1_acc']}; "
                    f"view1_nmi:{metric['view1_nmi']}; "
                    f"view2_acc:{metric['view2_acc']}; "
                    f"view2_nmi:{metric['view2_nmi']}"
                )
        print(run_str)
        append_run_log(run_paths.training_log, run_str.split(';'))
        output_hashes = {
            name: _file_sha256(path)
            for name, path in {
                'artifact': run_paths.artifact,
                'assignments': run_paths.assignments,
                'pretrain_view1': run_paths.pretrain_view1,
                'pretrain_view2': run_paths.pretrain_view2,
                'final_view1': run_paths.final_view1,
                'final_view2': run_paths.final_view2,
                'training_log': run_paths.training_log,
            }.items()
        }
        write_run_manifest(
            run_paths.manifest,
            {
                **manifest_base,
                'status': 'complete',
                'runtime_seconds': time.time() - time_start,
                'finished_at_utc': datetime.now(UTC).isoformat(),
                'metrics': metrics_payload,
                'output_sha256': output_hashes,
                'error': None,
            },
        )

    if ds_name in UNLABELED_DATASETS:
        avg_silhouette = float(
            np.nanmean([metrics["silhouette"] for metrics in run_metrics])
        )
        avg_str = (
            f'average over {args.runs} runs; silhouette:{avg_silhouette:.5f}; '
            f'protocol_id:{protocol.protocol_id}; '
            f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
            f'greedy_target_mode:{args.greedy_target_mode}; '
            f'time:{time.time() - time_all_start:.3f}'
        )
    else:
        avg_acc = float(np.nanmean([metrics["acc"] for metrics in run_metrics]))
        avg_nmi = float(np.nanmean([metrics["nmi"] for metrics in run_metrics]))
        avg_str = (
            f'average over {args.runs} runs; acc:{avg_acc:.5f}; nmi:{avg_nmi:.5f}; '
            f'protocol_id:{protocol.protocol_id}; '
            f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
            f'greedy_target_mode:{args.greedy_target_mode}; '
            f'time:{time.time() - time_all_start:.3f}'
    )
    print(avg_str)
    for config_dir in sorted(config_dirs):
        manifests = load_run_manifests(config_dir)
        runs_frame, summary_frame = build_summary_frames(manifests)
        write_public_frame(runs_frame, config_dir / 'mvdec_runs.csv')
        write_public_frame(summary_frame, config_dir / 'mvdec_summary.csv')
