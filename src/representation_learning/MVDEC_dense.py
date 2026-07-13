import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from utils import get_ACC_NMI, get_xy, log_csv

# Air pollution dataset (data/preprocessed_data/data_demvk.csv): 13 features,
# n_clusters=4 confirmed via the paper's own Elbow analysis. Fig. 1/Fig. 2
# produce a 23-wide output per view: 10 learned features + 13 reconstruction
# features for both views.
ds_name = 'AIRPOLLUTION'
input_shape = 13
hidden_units = 10
n_clusters = 4
view1_filters = [500, 500, 2000]
view2_base_units = 64
pretrain_epochs = 200
batch_size = 256
update_interval = 10
assignment_change_tolerance = 0.001
AIRPOLLUTION_ARTIFACT_PATH = (
    'data/preprocessed_data/airpollution_demvk_fused_representation.pkl'
)
TIKI_ARTIFACT_PATH = 'data/preprocessed_data/tiki_mvdec_fused_representation.pkl'
FINAL_TRAINING_OBJECTIVE = 'dekm2021_greedy_cluster_loss_after_reconstruction_pretrain'
UNLABELED_DATASETS = {
    'AIRPOLLUTION': {
        'csv_path': 'data/preprocessed_data/data_demvk.csv',
        'input_shape': 13,
        'hidden_units': 10,
        'view1_filters': [500, 500, 2000],
        'view2_base_units': 64,
        'n_clusters': 4,
        'artifact_path': AIRPOLLUTION_ARTIFACT_PATH,
    },
    'TIKI': {
        'csv_path': 'data/preprocessed_data/tiki_preprocessed.csv',
        'input_shape': 7,
        'hidden_units': 5,
        'view1_filters': [250, 250, 1000],
        'view2_base_units': 32,
        'n_clusters': 5,
        'artifact_path': TIKI_ARTIFACT_PATH,
    },
}


def view_output_width():
    return hidden_units + input_shape


def view_output_layout():
    return f'eq5_compatible_{hidden_units}_plus_{input_shape}'


def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def get_x_airpollution(
    dir_path=r'data/preprocessed_data/',
    log_print=True,
    shuffle_seed=None,
):
    # Separate from utils.py::get_xy (which already covers REUTERS/20NEWS/RCV1
    # as-is) because that dataset registry has no AIRPOLLUTION entry, and
    # air pollution has no labels to return alongside x.
    df = pd.read_csv(dir_path + 'data_demvk.csv')
    x = df.values.astype(np.float32)
    if shuffle_seed is None:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    if log_print:
        print(ds_name)
    # idx is returned so callers can map shuffled rows (and later, the fused
    # embedding/cluster assignment) back to the original CSV row order.
    return x, idx, list(df.columns)


def get_x_unlabeled_csv(csv_path, log_print=True, shuffle_seed=None):
    df = pd.read_csv(csv_path)
    x = df.values.astype(np.float32)
    if shuffle_seed is None:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    if log_print:
        print(ds_name)
    return x, idx, list(df.columns)


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
):
    # Fig. 1/Fig. 2 contract: h_fused is the average of both view outputs,
    # then K-means is applied to that fused representation.
    h_view1 = _restore_original_order(np.asarray(h_view1), orig_idx)
    h_view2 = _restore_original_order(np.asarray(h_view2), orig_idx)
    h_fused = _restore_original_order(np.asarray(h_fused), orig_idx)
    labels = _restore_original_order(np.asarray(labels), orig_idx)
    view_concat_representation = np.concatenate([h_view1, h_view2], axis=1)

    artifact = {
        'algorithm': 'MvDEC',
        'dataset': ds_name,
        'paper': '2025_Multi-view Deep Embedded Clustering',
        'fusion_contract': 'mvdec2025_figure_output_average',
        'view_output_layout': view_output_layout(),
        'final_training_objective': FINAL_TRAINING_OBJECTIVE,
        'h_view1': h_view1,
        'h_view2': h_view2,
        'h_fused': h_fused,
        'view_concat_representation': view_concat_representation,
        'labels': labels,
        'init': 'k-means',
        'score': float(score),
        'iteration': int(iteration),
        'input_dim': int(input_shape),
        'view1_latent_dim': int(hidden_units),
        'view2_latent_dim': int(hidden_units),
        'fusion_dim': int(h_fused.shape[1]),
        'n_clusters': int(n_clusters),
        'n_samples': int(h_fused.shape[0]),
        'feature_columns': feature_columns,
        'config': {
            'seed': random_seed,
            'n_clusters': int(n_clusters),
            'batch_size': int(batch_size),
            'pretrain_epochs': int(pretrain_epochs),
            'update_interval': int(update_interval),
            'assignment_change_tolerance': float(assignment_change_tolerance),
            'view1_latent_dim': int(hidden_units),
            'view2_latent_dim': int(hidden_units),
            'view1_filters': list(view1_filters),
            'view2_base_units': int(view2_base_units),
            'view1_output_dim': int(h_view1.shape[1]),
            'view2_output_dim': int(h_view2.shape[1]),
            'view_output_dim': int(h_fused.shape[1]),
            'view_output_layout': view_output_layout(),
            'final_training_objective': FINAL_TRAINING_OBJECTIVE,
            'fusion': 'h_fused = (view1_output + view2_output) / 2',
        },
    }

    output_dir = os.path.dirname(artifact_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(artifact_path, 'wb') as file:
        pickle.dump(artifact, file)
    print(f'MvDEC artifact was saved to {artifact_path}')


def model_view1(load_weights=True):
    # Air pollution/text datasets use the DEKM-style 500-500-2000 stack; Tiki
    # uses the smaller 250-250-1000 stack from Clustering_English_ver02 Fig. 1.
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
    # U-Net-inspired autoencoder (second view). Air pollution uses the 64-base
    # stack from the 2025 MvDEC paper; Tiki uses the smaller 32-base stack from
    # Clustering_English_ver02 Fig. 1. The decoder starts from h so
    # reconstruction pretraining updates the learned representation.
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
    skip1 = layers.Dense(b // 2, activation=activation, kernel_initializer=init)(e1)

    h = layers.Dense(
        hidden_units,
        activation=output_activation,
        kernel_initializer=init,
    )(bottleneck)

    x = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(h)
    x = layers.Concatenate()([x, e3])
    x = layers.Dense(8 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e2])
    x = layers.Dense(4 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e1])
    x = layers.Dense(2 * b, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(b, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, skip1])
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


def loss_train_base(y_true, y_pred):
    # Fig. 1/Fig. 2 produce a 23-wide output per view. The trailing 13 values
    # are trained against the input reconstruction target so Eq. 5 remains
    # applicable while the full 23-wide output is still available for fusion.
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, -input_shape:]
    return losses.mse(y_true, y_pred)


def train_base_view1(ds_xx):
    model = model_view1(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
    model.save_weights(f'weight_base_view1_{ds_name}.weights.h5')


def train_base_view2(ds_xx):
    model = model_view2(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
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
        acc, nmi = get_ACC_NMI(np.array(y), np.array(labels))
        return f'acc, nmi = {acc, nmi}', (acc, nmi)
    silhouette = silhouette_score(features, labels)
    return f'silhouette = {silhouette}', silhouette


def _cluster_sizes(labels):
    return np.bincount(np.asarray(labels), minlength=n_clusters).tolist()


def _log_training_phase(
    phase,
    space,
    metric_str,
    loss,
    n_change_assignment,
    labels,
    train_start_time,
    raw_reference=None,
):
    raw_metric_str = ''
    if raw_reference is not None:
        raw_silhouette = silhouette_score(raw_reference, labels)
        raw_metric_str = f' raw_silhouette_same_labels:{raw_silhouette};'
    log_str = (
        f'phase:{phase}; space:{space}; {metric_str}; loss:{loss}; '
        f'n_changed_assignment:{n_change_assignment}; '
        f'cluster_sizes:{_cluster_sizes(labels)};{raw_metric_str} '
        f'time:{time.time() - train_start_time:.3f}'
    )
    print(log_str)
    log_csv(log_str.split(';'), file_name=ds_name)


def train(
    x,
    y=None,
    orig_idx=None,
    feature_columns=None,
    random_seed=None,
    time_start=None,
    artifact_path=AIRPOLLUTION_ARTIFACT_PATH,
):
    # y is only available for labeled benchmark datasets (REUTERS/20NEWS/RCV1);
    # air pollution has no ground truth, so it stays None and silhouette is
    # used instead of ACC/NMI. orig_idx maps each (shuffled) row of x back to
    # its row number in the original, unshuffled source file -- needed so the
    # saved fused embedding/cluster assignment can later be joined back to the
    # original data for re-clustering or interpretation (e.g. land-use/traffic
    # correlation, as in the paper's Section 5.5).
    train_start_time = time.time() if time_start is None else time_start
    log_str = (
        'phase; space; metric; loss; n_changed_assignment; cluster_sizes; '
        'raw_silhouette_same_labels; '
        f'time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    )
    log_csv(log_str.split(';'), file_name=ds_name)
    model1 = model_view1()
    model2 = model_view2()

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    silhouette = np.nan
    acc = np.nan
    nmi = np.nan
    index = 0
    kmeans_n_init = 100
    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])
    raw_kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=kmeans_n_init,
        random_state=random_seed,
    ).fit(x)
    raw_metric_str, _ = _metric_for_labels(x, raw_kmeans.labels_, y=y)
    _log_training_phase(
        phase='baseline_raw_input',
        space='x',
        metric_str=raw_metric_str,
        loss=0.0,
        n_change_assignment=len(x),
        labels=raw_kmeans.labels_,
        train_start_time=train_start_time,
        raw_reference=x if y is None else None,
    )
    for ite in range(int(140 * 100)):
        if ite % update_interval == 0:
            h1 = model1(x).numpy()
            h2 = model2(x).numpy()
            H = (h1 + h2) / 2
            ans_kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=kmeans_n_init,
                random_state=random_seed,
            ).fit(H)
            kmeans_n_init = int(ans_kmeans.n_iter_ * 2)

            U = ans_kmeans.cluster_centers_
            assignment_new = ans_kmeans.labels_

            w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
            for i in range(len(assignment_new)):
                w[assignment_new[i], assignment[i]] += 1
            from scipy.optimize import linear_sum_assignment as linear_assignment
            ind = linear_assignment(-w)
            temp = np.array(assignment)
            for i in range(n_clusters):
                assignment[temp == ind[1][i]] = i
            n_change_assignment = np.sum(assignment_new != assignment)
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
            #
            loss = np.round(np.mean(loss_value), 5)
            metric_str, metric_value = _metric_for_labels(H, assignment, y=y)
            if y is not None:
                acc, nmi = metric_value
            else:
                silhouette = metric_value
            phase = (
                'baseline_pretrain_fused'
                if ite == 0
                else f'refine_iter_{ite // update_interval}'
            )
            _log_training_phase(
                phase=phase,
                space='H_fused',
                metric_str=metric_str,
                loss=loss,
                n_change_assignment=n_change_assignment,
                labels=assignment,
                train_start_time=train_start_time,
                raw_reference=x if y is None else None,
            )

        if n_change_assignment <= len(x) * assignment_change_tolerance:
            model1.save_weights(f'weight_final_view1_{ds_name}.weights.h5')
            model2.save_weights(f'weight_final_view2_{ds_name}.weights.h5')
            print('end')
            break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        y_true = H_vt[idx]
        temp = assignment[idx]
        for i in range(len(idx)):
            y_true[i, -1] = U_vt[temp[i], -1]

        with tf.GradientTape() as tape:
            y_pred1 = model1(x[idx])
            y_pred2 = model2(x[idx])
            h_pred = (y_pred1 + y_pred2) / 2
            y_pred_cluster = tf.matmul(h_pred, V)
            # After reconstruction pretraining, follow DEKM 2021: optimize only
            # the greedy clustering objective instead of carrying reconstruction
            # loss into the final representation update.
            loss_value = losses.mse(y_true, y_pred_cluster)
        trainable_variables = model1.trainable_variables + model2.trainable_variables
        grads = tape.gradient(loss_value, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables, strict=False))

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    if y is None and orig_idx is not None:
        # Save the final fused embedding + cluster assignment so clustering
        # can be redone (e.g. with a different k) or analyzed (e.g. joined
        # back to the original CSV via orig_index) without retraining.
        h1 = model1(x).numpy()
        h2 = model2(x).numpy()
        H = (h1 + h2) / 2
        assignment = KMeans(
            n_clusters=n_clusters,
            n_init=kmeans_n_init,
            random_state=random_seed,
        ).fit(H).labels_
        silhouette = silhouette_score(H, assignment)
        metric_str = f'silhouette = {silhouette}'
        _log_training_phase(
            phase='final_recomputed_artifact',
            space='H_fused',
            metric_str=metric_str,
            loss=np.round(np.mean(loss_value), 5),
            n_change_assignment=0,
            labels=assignment,
            train_start_time=train_start_time,
            raw_reference=x,
        )
        if not os.path.exists('output'):
            os.makedirs('output')
        h_ordered = _restore_original_order(H, orig_idx)
        labels_ordered = _restore_original_order(assignment, orig_idx)
        result = pd.DataFrame(
            h_ordered,
            columns=[f'h_{i}' for i in range(view_output_width())],
        )
        result.insert(0, 'orig_index', np.arange(len(orig_idx)))
        result['cluster'] = labels_ordered
        result.to_csv(f'output/{ds_name}_clusters.csv', index=False)
        save_airpollution_mvdec_artifact(
            artifact_path=artifact_path,
            h_view1=h1,
            h_view2=h2,
            h_fused=H,
            labels=assignment,
            score=silhouette,
            iteration=ite // update_interval,
            orig_idx=orig_idx,
            feature_columns=feature_columns,
            random_seed=random_seed,
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
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
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
    update_interval = 10
    assignment_change_tolerance = 0.001
    if (
        ds_name in UNLABELED_DATASETS
        and args.artifact_path == AIRPOLLUTION_ARTIFACT_PATH
    ):
        args.artifact_path = UNLABELED_DATASETS[ds_name]['artifact_path']

    run_metrics = []
    time_all_start = time.time()
    for run_index in range(args.runs):
        run_seed = None if args.seed is None else args.seed + run_index
        set_random_seed(run_seed)
        time_start = time.time()
        orig_idx = None
        feature_columns = None
        if ds_name in UNLABELED_DATASETS:
            x, orig_idx, feature_columns = get_x_unlabeled_csv(
                UNLABELED_DATASETS[ds_name]['csv_path'],
                shuffle_seed=run_seed,
            )
            y = None
        else:
            x, y = get_xy(
                ds_name=ds_name,
                dir_path=r'external_repos/DEKM/datasets/',
                shuffle_seed=run_seed,
            )
        ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(
            8000, seed=run_seed
        ).batch(pretrain_batch_size)
        train_base_view1(ds_xx)
        train_base_view2(ds_xx)
        metric = train(
            x,
            y=y,
            orig_idx=orig_idx,
            feature_columns=feature_columns,
            random_seed=run_seed,
            time_start=time_start,
            artifact_path=args.artifact_path,
        )
        run_metrics.append(metric)
        if y is None:
            run_str = (
                f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
                f'silhouette:{metric}; time:{time.time() - time_start:.3f}'
            )
        else:
            acc, nmi = metric
            run_str = (
                f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
                f'acc:{acc}; nmi:{nmi}; time:{time.time() - time_start:.3f}'
            )
        print(run_str)
        log_csv(run_str.split(';'), file_name=ds_name)

    if ds_name in UNLABELED_DATASETS:
        avg_silhouette = float(np.nanmean(np.asarray(run_metrics, dtype=float)))
        avg_str = (
            f'average over {args.runs} runs; silhouette:{avg_silhouette:.5f}; '
            f'time:{time.time() - time_all_start:.3f}'
        )
    else:
        metrics = np.asarray(run_metrics, dtype=float)
        avg_acc = float(np.nanmean(metrics[:, 0]))
        avg_nmi = float(np.nanmean(metrics[:, 1]))
        avg_str = (
            f'average over {args.runs} runs; acc:{avg_acc:.5f}; nmi:{avg_nmi:.5f}; '
            f'time:{time.time() - time_all_start:.3f}'
        )
    print(avg_str)
    log_csv(avg_str.split(';'), file_name=ds_name)
