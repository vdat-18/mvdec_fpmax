import argparse
import hashlib
import time
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
from pipeline.external_metrics import compute_external_metrics
from pipeline.public_artifacts import (
    PUBLIC_ARTIFACT_SCHEMA_VERSION,
    public_assignment_frame,
    public_run_stem,
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
assignment_change_tolerance = 0.01
# DEKM 2021 (Fig. 4) shows pulling every embedding dimension toward the
# centroids underperforms the greedy single-direction pull, so the L2 K-means
# term defaults to 0; the literal MvDEC 2025 Eq. 11 stays available via
# --lambda-kmeans 1.
DEFAULT_LAMBDA_KMEANS = 0.0
DEFAULT_LAMBDA_GREEDY = 1.0
lambda_reconstruction = 1.0
lambda_kmeans = DEFAULT_LAMBDA_KMEANS
lambda_orthonormal = 0.0
lambda_greedy = DEFAULT_LAMBDA_GREEDY
AIRPOLLUTION_ARTIFACT_PATH = (
    'data/preprocessed_data/airpollution_demvk_fused_representation.pkl'
)
TIKI_ARTIFACT_PATH = 'data/preprocessed_data/tiki_mvdec_fused_representation.pkl'
GREEDY_EIGEN_DIRECTIONS = ('largest', 'smallest')
GREEDY_TARGET_MODES = ('selected_dimension_only', 'frozen_snapshot')
DEFAULT_GREEDY_EIGEN_DIRECTION = 'largest'
DEFAULT_GREEDY_TARGET_MODE = 'selected_dimension_only'
KMEANS_N_INIT = 100
KMEANS_REFRESH_POLICY = 'one_epoch'
REFINEMENT_BATCHING_POLICY = 'balanced_shuffled_each_epoch'
MAX_REFINEMENT_EPOCHS = 1400
TRAINING_STOP_REASONS = ('converged_assignment', 'max_epochs_reached')
UNLABELED_DATASETS = {
    'AIRPOLLUTION': {
        'csv_path': 'data/preprocessed_data/data_demvk.csv',
        'input_shape': 13,
        'hidden_units': 10,
        'view1_filters': [500, 500, 2000],
        'view2_base_units': 64,
        'n_clusters': 4,
        'artifact_path': AIRPOLLUTION_ARTIFACT_PATH,
        'scaling_method': 'minmax',
    },
    'TIKI': {
        'csv_path': 'data/preprocessed_data/tiki_preprocessed.csv',
        'input_shape': 7,
        'hidden_units': 5,
        'view1_filters': [250, 250, 1000],
        'view2_base_units': 32,
        'n_clusters': 5,
        'artifact_path': TIKI_ARTIFACT_PATH,
        'scaling_method': 'none',
    },
}


def view_output_width():
    return hidden_units + input_shape


def view_output_layout():
    return f'eq5_compatible_{hidden_units}_plus_{input_shape}'


def final_training_objective(kmeans_weight, greedy_weight):
    parts = ['reconstruction']
    if kmeans_weight > 0:
        parts.append('kmeans')
    if greedy_weight > 0:
        parts.append('greedy')
    return f'mvdec2025_latent_joint_{"_".join(parts)}_l3_trace_logged'


def latent_embedding(view_output):
    return view_output[:, :hidden_units]


def reconstruction_output(view_output):
    return view_output[:, -input_shape:]


def fused_latent_embedding(view1_output, view2_output):
    return (latent_embedding(view1_output) + latent_embedding(view2_output)) / 2


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


def append_artifact_suffix(artifact_path, suffix):
    if not suffix:
        return str(artifact_path)
    path = Path(artifact_path)
    return str(path.with_name(f'{path.stem}{suffix}{path.suffix}'))


def loss_ablation_tag(kmeans_weight, greedy_weight):
    if (
        kmeans_weight == DEFAULT_LAMBDA_KMEANS
        and greedy_weight == DEFAULT_LAMBDA_GREEDY
    ):
        return ''
    return f'_l2w{kmeans_weight:g}_l4w{greedy_weight:g}'


def artifact_path_for_greedy_mode(artifact_path, direction, target_mode):
    greedy_eigen_index(direction)
    validate_greedy_target_mode(target_mode)
    if (
        direction == DEFAULT_GREEDY_EIGEN_DIRECTION
        and target_mode == DEFAULT_GREEDY_TARGET_MODE
    ):
        return str(artifact_path)
    return append_artifact_suffix(
        artifact_path,
        f'_{direction}_eigen_{target_mode}',
    )


def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def make_kmeans(random_seed):
    return KMeans(
        n_clusters=n_clusters,
        n_init=KMEANS_N_INIT,
        random_state=random_seed,
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


def validate_max_refinement_epochs(max_refinement_epochs):
    if max_refinement_epochs < 1:
        raise ValueError('max_refinement_epochs must be at least 1.')
    return int(max_refinement_epochs)


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
    return raw_x.astype(np.float32), x.astype(np.float32), metadata


def get_x_airpollution(
    dir_path=r'data/preprocessed_data/',
    log_print=True,
    shuffle_seed=None,
):
    return get_x_unlabeled_csv(
        Path(dir_path) / 'data_demvk.csv',
        log_print=log_print,
        shuffle_seed=shuffle_seed,
        scaling_method='minmax',
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
):
    greedy_eigen_index(greedy_eigen_direction)
    validate_greedy_target_mode(greedy_target_mode)
    validate_max_refinement_epochs(max_refinement_epochs)
    if stop_reason not in TRAINING_STOP_REASONS:
        raise ValueError(f'Unsupported training stop reason: {stop_reason!r}.')
    greedy_eigen_position = (
        hidden_units - 1 if greedy_eigen_direction == 'largest' else 0
    )
    training_objective = final_training_objective(lambda_kmeans, lambda_greedy)
    if ds_name == 'AIRPOLLUTION':
        if not isinstance(preprocessing_metadata, dict):
            raise ValueError(
                'Air Pollution artifacts require Min-Max preprocessing metadata.'
            )
        if preprocessing_metadata.get('method') != 'minmax':
            raise ValueError('Air Pollution artifacts require Min-Max scaled input.')
        if preprocessing_metadata.get('feature_columns') != feature_columns:
            raise ValueError(
                'Air Pollution scaler columns must match artifact feature columns.'
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

    artifact = {
        'schema_version': PUBLIC_ARTIFACT_SCHEMA_VERSION,
        'algorithm': 'MvDEC',
        'dataset': ds_name,
        'paper': '2025_Multi-view Deep Embedded Clustering',
        'fusion_contract': 'mvdec2025_encoder_average',
        'view_output_layout': view_output_layout(),
        'final_training_objective': training_objective,
        'h_view1': h_view1,
        'h_view2': h_view2,
        'h_fused': h_fused,
        'view_concat_representation': view_concat_representation,
        'labels': labels,
        'true_labels': true_labels_ordered,
        'row_indices': np.arange(len(labels), dtype=int),
        'source_sha256': source_sha256,
        'init': 'k-means',
        'score': float(score),
        'iteration': int(iteration),
        'input_dim': int(input_shape),
        'view1_latent_dim': int(hidden_units),
        'view2_latent_dim': int(hidden_units),
        'fusion_dim': int(h_fused.shape[1]),
        'n_clusters': int(n_clusters),
        'kmeans_n_init': int(KMEANS_N_INIT),
        'kmeans_refresh_policy': KMEANS_REFRESH_POLICY,
        'refinement_batching_policy': REFINEMENT_BATCHING_POLICY,
        'batches_per_epoch': int(batches_per_epoch),
        'stop_reason': stop_reason,
        'refinement_epochs_completed': int(refinement_epochs_completed),
        'n_samples': int(h_fused.shape[0]),
        'feature_columns': feature_columns,
        'eigenvalue_order': 'ascending',
        'greedy_eigen_direction': greedy_eigen_direction,
        'greedy_eigen_index': int(greedy_eigen_position),
        'greedy_target_mode': greedy_target_mode,
        'preprocessing': preprocessing_metadata,
        'config': {
            'seed': random_seed,
            'n_clusters': int(n_clusters),
            'kmeans_n_init': int(KMEANS_N_INIT),
            'batch_size': int(batch_size),
            'pretrain_epochs': int(pretrain_epochs),
            'kmeans_refresh_policy': KMEANS_REFRESH_POLICY,
            'refinement_batching_policy': REFINEMENT_BATCHING_POLICY,
            'batches_per_epoch': int(batches_per_epoch),
            'update_interval': int(batches_per_epoch),
            'max_refinement_epochs': int(max_refinement_epochs),
            'max_training_steps': int(max_refinement_epochs * batches_per_epoch),
            'stop_reason': stop_reason,
            'refinement_epochs_completed': int(refinement_epochs_completed),
            'assignment_change_tolerance': float(assignment_change_tolerance),
            'view1_latent_dim': int(hidden_units),
            'view2_latent_dim': int(hidden_units),
            'view1_filters': list(view1_filters),
            'view2_base_units': int(view2_base_units),
            'view1_embedding_dim': int(h_view1.shape[1]),
            'view2_embedding_dim': int(h_view2.shape[1]),
            'fusion_dim': int(h_fused.shape[1]),
            'model_view_output_dim': int(view_output_width()),
            'model_view_output_layout': view_output_layout(),
            'final_training_objective': training_objective,
            'eigenvalue_order': 'ascending',
            'greedy_eigen_direction': greedy_eigen_direction,
            'greedy_eigen_index': int(greedy_eigen_position),
            'greedy_target_mode': greedy_target_mode,
            'loss_weights': {
                'reconstruction': float(lambda_reconstruction),
                'kmeans': float(lambda_kmeans),
                'orthonormal_trace_logged': float(lambda_orthonormal),
                'greedy': float(lambda_greedy),
            },
            'fusion': 'h_fused = (view1_latent + view2_latent) / 2',
            'preprocessing': preprocessing_metadata,
        },
    }
    if external_metrics is not None:
        artifact['acc'] = float(external_metrics.acc)
        artifact['nmi'] = float(external_metrics.nmi)

    write_public_artifact(artifact, Path(artifact_path))
    print(f'MvDEC artifact was saved to {artifact_path}')


def model_view1(load_weights=True):
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
        model.load_weights(f'weight_base_view1_{ds_name}.weights.h5')
        print('model_view1: weights was loaded')
    return model


def model_view2(load_weights=True):
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

    h = layers.Dense(
        hidden_units,
        activation=output_activation,
        kernel_initializer=init,
    )(bottleneck)

    x = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(h)
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
    y = layers.Dense(
        input_shape,
        activation=output_activation,
        kernel_initializer=init,
    )(x)

    output = layers.Concatenate()([h, y])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        model.load_weights(f'weight_base_view2_{ds_name}.weights.h5')
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


def train_base_view1(ds_xx):
    model = model_view1(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    history = model.fit(ds_xx, epochs=pretrain_epochs, verbose=0)
    print(
        f'pretrain view1 {ds_name}: epochs={pretrain_epochs}; '
        f'final_loss={history.history["loss"][-1]:.5f}'
    )
    model.save_weights(f'weight_base_view1_{ds_name}.weights.h5')


def train_base_view2(ds_xx):
    model = model_view2(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    history = model.fit(ds_xx, epochs=pretrain_epochs, verbose=0)
    print(
        f'pretrain view2 {ds_name}: epochs={pretrain_epochs}; '
        f'final_loss={history.history["loss"][-1]:.5f}'
    )
    model.save_weights(f'weight_base_view2_{ds_name}.weights.h5')


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


def recompute_final_clustering(model1, model2, x, random_seed):
    view1_output = model1(x).numpy()
    view2_output = model2(x).numpy()
    h1 = latent_embedding(view1_output)
    h2 = latent_embedding(view2_output)
    h_fused = (h1 + h2) / 2
    labels = make_kmeans(random_seed).fit(h_fused).labels_
    return h1, h2, h_fused, labels


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
        print(log_str)
    log_csv(log_str.split(';'), file_name=ds_name if file_name is None else file_name)


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
    artifact_path=AIRPOLLUTION_ARTIFACT_PATH,
    source_sha256=None,
):
    train_start_time = time.time() if time_start is None else time_start
    greedy_index = greedy_eigen_index(greedy_eigen_direction)
    validate_greedy_target_mode(greedy_target_mode)
    max_refinement_epochs = validate_max_refinement_epochs(
        max_refinement_epochs
    )
    batches_per_epoch = number_of_batches(len(x), batch_size)
    kmeans_refresh_interval = batches_per_epoch
    max_training_steps = max_refinement_epochs * batches_per_epoch
    batch_rng = np.random.default_rng(random_seed)
    experiment_tag = (
        f'{greedy_eigen_direction}_eigen_{greedy_target_mode}'
        f'{loss_ablation_tag(lambda_kmeans, lambda_greedy)}'
    )
    train_log_name = f'{ds_name}_{experiment_tag.upper()}'
    log_str = (
        'phase; space; metric; silhouette_input_space; loss; '
        'n_changed_assignment; cluster_sizes; '
        'greedy_eigen_direction; greedy_eigen_index; greedy_eigenvalue; '
        'greedy_target_mode; '
        f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    )
    log_csv(log_str.split(';'), file_name=train_log_name)
    model1 = model_view1()
    model2 = model_view2()

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
    stop_reason = 'max_epochs_reached'
    refinement_epochs_completed = 0
    epoch_batch_losses = []
    last_epoch_loss_means = None
    assignment = np.array([-1] * len(x))
    epoch_batches = None
    if raw_x is not None and raw_x.shape != x.shape:
        raise ValueError('raw_x and x must have the same shape.')

    if raw_x is not None:
        raw_kmeans = make_kmeans(random_seed).fit(raw_x)
        raw_metric_str, _ = _metric_for_labels(raw_x, raw_kmeans.labels_, y=y)
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
        )

    input_kmeans = make_kmeans(random_seed).fit(x)
    input_metric_str, _ = _metric_for_labels(x, input_kmeans.labels_, y=y)
    input_phase = 'baseline_scaled_input' if raw_x is not None else 'baseline_raw_input'
    input_space = 'x_minmax' if raw_x is not None else 'x'
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
    )
    for ite in range(max_training_steps):
        log_paper_step_checkpoint = ite % kmeans_refresh_interval == 0
        log_step9_end_of_epoch = is_refinement_epoch_end(
            ite,
            kmeans_refresh_interval,
        )
        if log_paper_step_checkpoint:
            epoch_batch_losses = []
            epoch_batches = epoch_batch_indices(len(x), batch_size, batch_rng)
            view1_output = model1(x).numpy()
            view2_output = model2(x).numpy()
            H = fused_latent_embedding(view1_output, view2_output)
            ans_kmeans = make_kmeans(random_seed).fit(H)

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
                'greedy_eigen_direction': greedy_eigen_direction,
                'greedy_eigen_index': selected_eigen_position,
                'greedy_eigenvalue': selected_eigenvalue,
                'greedy_target_mode': greedy_target_mode,
                'kmeans_n_init': KMEANS_N_INIT,
                'kmeans_refresh_policy': KMEANS_REFRESH_POLICY,
                'refinement_batching_policy': REFINEMENT_BATCHING_POLICY,
                'batches_per_epoch': batches_per_epoch,
                'lambda_kmeans': lambda_kmeans,
                'lambda_greedy': lambda_greedy,
            }
            loss = np.round(_loss_scalar(loss_value), 5)
            metric_str, metric_value = _metric_for_labels(H, assignment, y=y)
            if y is not None:
                acc, nmi = metric_value
            else:
                silhouette = metric_value
            paper_step_suffix = (
                'pre_refine'
                if ite == 0
                else f'refine_epoch_{ite // kmeans_refresh_interval}'
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
            )
            greedy_metric_str, _ = _metric_for_labels(
                H_vt_greedy_target,
                assignment,
                y=y,
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
                print_console=False,
            )

        if n_change_assignment <= len(x) * assignment_change_tolerance:
            stop_reason = 'converged_assignment'
            refinement_epochs_completed = ite // kmeans_refresh_interval
            break
        idx = epoch_batches[index]
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
            h_pred = fused_latent_embedding(y_pred1, y_pred2)
            y_pred_cluster = tf.matmul(h_pred, V_tensor)
            orthonormal_residual = tf.matmul(h_pred - U_batch_tensor, V_tensor)
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
            if greedy_target_mode == 'selected_dimension_only':
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
            )
            after_update_metric_str, _ = _metric_for_labels(
                H_after_update,
                assignment,
                y=y,
            )
            _log_training_phase(
                phase=f'paper_step9_after_full_epoch_{refinement_epoch}',
                space='H_fused_latent_same_labels',
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
                    'refinement_epoch': refinement_epoch,
                    'step9_metric_scope': 'full_dataset_after_epoch',
                    'step9_loss_scope': 'epoch_mean',
                    'step9_loss_batches': len(epoch_batch_losses),
                    'step9_loss_samples': sum(
                        losses['sample_count'] for losses in epoch_batch_losses
                    ),
                },
                file_name=train_log_name,
                print_console=False,
            )

        index = (index + 1) % batches_per_epoch

    if stop_reason == 'max_epochs_reached':
        refinement_epochs_completed = max_refinement_epochs
    model1.save_weights(
        f'weight_final_view1_{ds_name}_{experiment_tag}.weights.h5'
    )
    model2.save_weights(
        f'weight_final_view2_{ds_name}_{experiment_tag}.weights.h5'
    )
    stop_log_str = (
        f'phase:training_stop; stop_reason:{stop_reason}; '
        f'refinement_epochs_completed:{refinement_epochs_completed}; '
        f'max_refinement_epochs:{max_refinement_epochs}; '
        f'time:{time.time() - train_start_time:.3f}'
    )
    print(stop_log_str)
    log_csv(stop_log_str.split(';'), file_name=train_log_name)

    h1, h2, H, final_assignment = recompute_final_clustering(
        model1,
        model2,
        x,
        random_seed,
    )
    final_n_change_assignment = count_aligned_assignment_changes(
        assignment,
        final_assignment,
    )
    assignment = final_assignment
    metric_str, final_metric = _metric_for_labels(H, assignment, y=y)
    artifact_score = float(silhouette_score(H, assignment))
    external_metrics = None
    if y is None:
        silhouette = final_metric
    else:
        acc, nmi = final_metric
        external_metrics = compute_external_metrics(
            y,
            assignment,
            n_clusters=n_clusters,
        )
    final_phase = (
        'final_recomputed_artifact'
        if y is None and orig_idx is not None
        else 'final_recomputed_state'
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
        extra_fields={
            'greedy_eigen_direction': greedy_eigen_direction,
            'greedy_eigen_index': (
                hidden_units - 1 if greedy_eigen_direction == 'largest' else 0
            ),
            'greedy_target_mode': greedy_target_mode,
            'lambda_kmeans': lambda_kmeans,
            'lambda_greedy': lambda_greedy,
            'kmeans_refresh_policy': KMEANS_REFRESH_POLICY,
            'batches_per_epoch': batches_per_epoch,
            'stop_reason': stop_reason,
            'refinement_epochs_completed': refinement_epochs_completed,
            'max_refinement_epochs': max_refinement_epochs,
            'loss_scope': 'last_completed_epoch_mean',
        },
        file_name=train_log_name,
    )

    if orig_idx is not None:
        h_ordered = _restore_original_order(H, orig_idx)
        labels_ordered = _restore_original_order(assignment, orig_idx)
        artifact_path = Path(artifact_path)
        if y is None:
            result = pd.DataFrame(
                h_ordered,
                columns=[f'h_{i}' for i in range(H.shape[1])],
            )
            result.insert(0, 'orig_index', np.arange(len(orig_idx)))
            result['cluster'] = labels_ordered
            assignments_path = Path('output') / (
                f'{ds_name}_{experiment_tag}_clusters.csv'
            )
        else:
            result = public_assignment_frame(
                dataset=ds_name,
                method='MvDEC',
                labels=labels_ordered,
                true_labels=_restore_original_order(y, orig_idx),
                random_seed=random_seed,
            )
            assignments_path = artifact_path.with_name(
                f'{artifact_path.stem}_assignments.csv'
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
            iteration=refinement_epochs_completed,
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
        )

    if y is not None:
        return acc, nmi
    return silhouette


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='select dataset:AIRPOLLUTION,TIKI,REUTERS,20NEWS,RCV1',
    )
    parser.add_argument('ds_name', default='AIRPOLLUTION')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--artifact-path', default=AIRPOLLUTION_ARTIFACT_PATH)
    parser.add_argument('--dataset-root', type=Path, default=DEKM_DATASET_DIR)
    parser.add_argument(
        '--public-output-dir',
        type=Path,
        default=PUBLIC_BENCHMARK_OUTPUT_DIR / 'mvdec',
    )
    parser.add_argument(
        '--max-refinement-epochs',
        type=int,
        default=MAX_REFINEMENT_EPOCHS,
        help='Safety cap for full-epoch refinement cycles.',
    )
    parser.add_argument(
        '--greedy-eigen-direction',
        choices=GREEDY_EIGEN_DIRECTIONS,
        default=DEFAULT_GREEDY_EIGEN_DIRECTION,
        help='Use the largest or smallest within-cluster scatter eigenvalue for L4.',
    )
    parser.add_argument(
        '--greedy-target-mode',
        choices=GREEDY_TARGET_MODES,
        default=DEFAULT_GREEDY_TARGET_MODE,
        help='Use the paper-style selected dimension or the frozen DEKM target.',
    )
    parser.add_argument(
        '--lambda-kmeans',
        type=float,
        default=DEFAULT_LAMBDA_KMEANS,
        help=(
            'Weight of the L2 K-means loss; defaults to 0 per DEKM 2021 '
            'Fig. 4, set 1 for the literal MvDEC 2025 Eq. 11.'
        ),
    )
    parser.add_argument(
        '--lambda-greedy',
        type=float,
        default=DEFAULT_LAMBDA_GREEDY,
        help='Weight of the L4 greedy loss; set 0 to ablate it.',
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.lambda_kmeans < 0 or args.lambda_greedy < 0:
        raise ValueError('--lambda-kmeans and --lambda-greedy must be non-negative.')
    lambda_kmeans = args.lambda_kmeans
    lambda_greedy = args.lambda_greedy
    validate_max_refinement_epochs(args.max_refinement_epochs)
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
    assignment_change_tolerance = 0.01
    if (
        ds_name in UNLABELED_DATASETS
        and args.artifact_path == AIRPOLLUTION_ARTIFACT_PATH
    ):
        args.artifact_path = append_artifact_suffix(
            artifact_path_for_greedy_mode(
                UNLABELED_DATASETS[ds_name]['artifact_path'],
                args.greedy_eigen_direction,
                args.greedy_target_mode,
            ),
            loss_ablation_tag(lambda_kmeans, lambda_greedy),
        )

    public_dataset = (
        None
        if ds_name in UNLABELED_DATASETS
        else load_dekm_public_dataset(ds_name, args.dataset_root)
    )
    run_metrics = []
    public_run_records = []
    run_experiment_tag = (
        f'{args.greedy_eigen_direction}_eigen_{args.greedy_target_mode}'
        f'{loss_ablation_tag(lambda_kmeans, lambda_greedy)}'
    )
    run_log_name = f'{ds_name}_{run_experiment_tag.upper()}'
    time_all_start = time.time()
    for run_index in range(args.runs):
        run_seed = None if args.seed is None else args.seed + run_index
        set_random_seed(run_seed)
        time_start = time.time()
        orig_idx = None
        feature_columns = None
        raw_x = None
        preprocessing_metadata = None
        source_sha256 = None
        run_artifact_path = args.artifact_path
        if ds_name in UNLABELED_DATASETS:
            dataset_config = UNLABELED_DATASETS[ds_name]
            x, orig_idx, feature_columns, source_x, preprocessing_metadata = (
                get_x_unlabeled_csv(
                    dataset_config['csv_path'],
                    shuffle_seed=run_seed,
                    scaling_method=dataset_config['scaling_method'],
                    include_preprocessing=True,
                )
            )
            if preprocessing_metadata['method'] != 'none':
                raw_x = source_x
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
            run_stem = public_run_stem('MvDEC', ds_name, run_seed, run_index + 1)
            run_artifact_path = (
                args.public_output_dir / ds_name.lower() / f'{run_stem}.pkl'
            )
        ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(
            8000, seed=run_seed
        ).batch(pretrain_batch_size)
        train_base_view1(ds_xx)
        train_base_view2(ds_xx)
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
            artifact_path=run_artifact_path,
            source_sha256=source_sha256,
        )
        run_metrics.append(metric)
        if y is None:
            run_str = (
                f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
                f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
                f'greedy_target_mode:{args.greedy_target_mode}; '
                f'silhouette:{metric}; time:{time.time() - time_start:.3f}'
            )
        else:
            acc, nmi = metric
            public_run_records.append(
                {
                    'dataset': ds_name,
                    'method': 'MvDEC',
                    'run': run_index + 1,
                    'random_seed': run_seed,
                    'n_samples': len(public_dataset.X),
                    'n_features': public_dataset.X.shape[1],
                    'n_clusters': n_clusters,
                    'source_sha256': public_dataset.source_sha256,
                    'acc': acc,
                    'nmi': nmi,
                    'artifact_path': str(run_artifact_path),
                    'assignments_path': str(
                        run_artifact_path.with_name(
                            f'{run_artifact_path.stem}_assignments.csv'
                        )
                    ),
                    'fit_time_seconds': time.time() - time_start,
                }
            )
            run_str = (
                f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
                f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
                f'greedy_target_mode:{args.greedy_target_mode}; '
                f'acc:{acc}; nmi:{nmi}; time:{time.time() - time_start:.3f}'
            )
        print(run_str)
        log_csv(
            run_str.split(';'),
            file_name=run_log_name,
        )

    if ds_name in UNLABELED_DATASETS:
        avg_silhouette = float(np.nanmean(np.asarray(run_metrics, dtype=float)))
        avg_str = (
            f'average over {args.runs} runs; silhouette:{avg_silhouette:.5f}; '
            f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
            f'greedy_target_mode:{args.greedy_target_mode}; '
            f'time:{time.time() - time_all_start:.3f}'
        )
    else:
        metrics = np.asarray(run_metrics, dtype=float)
        avg_acc = float(np.nanmean(metrics[:, 0]))
        avg_nmi = float(np.nanmean(metrics[:, 1]))
        avg_str = (
            f'average over {args.runs} runs; acc:{avg_acc:.5f}; nmi:{avg_nmi:.5f}; '
            f'greedy_eigen_direction:{args.greedy_eigen_direction}; '
            f'greedy_target_mode:{args.greedy_target_mode}; '
            f'time:{time.time() - time_all_start:.3f}'
        )
    print(avg_str)
    log_csv(
        avg_str.split(';'),
        file_name=run_log_name,
    )
    if public_run_records:
        runs_frame = pd.DataFrame.from_records(public_run_records)
        summary_frame = pd.DataFrame(
            [
                {
                    'dataset': ds_name,
                    'method': 'MvDEC',
                    'requested_runs': len(runs_frame),
                    'acc_mean': runs_frame['acc'].mean(),
                    'acc_std': (
                        runs_frame['acc'].std(ddof=1)
                        if len(runs_frame) > 1
                        else 0.0
                    ),
                    'nmi_mean': runs_frame['nmi'].mean(),
                    'nmi_std': (
                        runs_frame['nmi'].std(ddof=1)
                        if len(runs_frame) > 1
                        else 0.0
                    ),
                    'source_sha256': public_dataset.source_sha256,
                }
            ]
        )
        output_dataset_dir = args.public_output_dir / ds_name.lower()
        write_public_frame(runs_frame, output_dataset_dir / 'mvdec_runs.csv')
        write_public_frame(summary_frame, output_dataset_dir / 'mvdec_summary.csv')
