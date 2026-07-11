import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from utils import get_ACC_NMI
from utils import get_xy
from utils import log_csv
import time
import argparse
import os

# Air pollution dataset (data/preprocessed_data/data_demvk.csv): 13 features,
# n_clusters=4 confirmed via the paper's own Elbow analysis; hidden_units=10
# matches model_view1's embedding width in Fig. 2 (both views must be equal
# width so Eq. 4's averaging fusion works). Only used to fully specify the
# two architectures below -- no training/data-loading wired up yet.
ds_name = 'AIRPOLLUTION'
input_shape = 13
hidden_units = 10
n_clusters = 4
pretrain_epochs = 200
batch_size = 256
update_interval = 10


def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def get_x_airpollution(dir_path=r'data/preprocessed_data/', log_print=True, shuffle_seed=None):
    # Separate from utils.py::get_xy (which already covers REUTERS/20NEWS/RCV1
    # as-is) because that dataset registry has no AIRPOLLUTION entry, and
    # air pollution has no labels to return alongside x.
    x = pd.read_csv(dir_path + 'data_demvk.csv').values.astype(np.float32)
    if shuffle_seed is None:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    if log_print:
        print(ds_name)
    # idx is returned so callers can map shuffled rows (and later, the fused
    # embedding/cluster assignment) back to the original CSV row order.
    return x, idx


def model_view1(load_weights=True):
    # d-500-500-2000-hidden_units, same architecture as DEKM_dense.py::model_conv
    filters = [500, 500, 2000]
    init = 'glorot_uniform'
    activation = 'relu'
    input = layers.Input(shape=(input_shape,))
    x = input
    for i in range(len(filters)):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(hidden_units, kernel_initializer=init)(x)
    h = x

    for i in range(len(filters) - 1, -1, -1):
        x = layers.Dense(filters[i], activation=activation, kernel_initializer=init)(x)
    y = layers.Dense(input_shape, kernel_initializer=init)(x)

    output = layers.Concatenate()([h, y])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        model.load_weights(f'weight_base_view1_{ds_name}.weights.h5')
        print('model_view1: weights was loaded')
    return model


def model_view2(load_weights=True):
    # U-Net-inspired autoencoder (second view), see docs/2025_Multi-view Deep
    # Embedded Clustering...pdf, Fig. 2. Layer widths and skip-connection
    # concat sizes below (768/384/192/96, matching 512+256, 256+128, 128+64,
    # 64+32) are read directly off Fig. 2. The embedding tap Dense(hidden_units)
    # right after the 1024-wide bottleneck is an inference, not a node we could
    # directly read in the figure: it is required so this view's embedding has
    # the same width as model_view1's (Eq. 4 fuses the two by averaging, which
    # only works if both are the same size). Verify against Fig. 2 if you have
    # a clearer look at that exact spot.
    init = 'glorot_uniform'
    activation = 'relu'
    input = layers.Input(shape=(input_shape,))

    e1 = layers.Dense(64, activation=activation, kernel_initializer=init)(input)
    e1 = layers.Dense(64, activation=activation, kernel_initializer=init)(e1)
    e2 = layers.Dense(128, activation=activation, kernel_initializer=init)(e1)
    e2 = layers.Dense(128, activation=activation, kernel_initializer=init)(e2)
    e3 = layers.Dense(256, activation=activation, kernel_initializer=init)(e2)
    e3 = layers.Dense(256, activation=activation, kernel_initializer=init)(e3)
    e4 = layers.Dense(512, activation=activation, kernel_initializer=init)(e3)
    e4 = layers.Dense(512, activation=activation, kernel_initializer=init)(e4)
    bottleneck = layers.Dense(1024, activation=activation, kernel_initializer=init)(e4)
    skip1 = layers.Dense(32, activation=activation, kernel_initializer=init)(e1)

    x = layers.Dense(hidden_units, kernel_initializer=init)(bottleneck)
    h = x

    x = layers.Dense(512, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e3])
    x = layers.Dense(512, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(256, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e2])
    x = layers.Dense(256, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(128, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, e1])
    x = layers.Dense(128, activation=activation, kernel_initializer=init)(x)
    x = layers.Dense(64, activation=activation, kernel_initializer=init)(x)
    x = layers.Concatenate()([x, skip1])
    x = layers.Dense(64, activation=activation, kernel_initializer=init)(x)
    y = layers.Dense(input_shape, kernel_initializer=init)(x)

    output = layers.Concatenate()([h, y])
    model = Model(inputs=input, outputs=output)
    if load_weights:
        model.load_weights(f'weight_base_view2_{ds_name}.weights.h5')
        print('model_view2: weights was loaded')
    return model


def loss_train_base(y_true, y_pred):
    # Shared by both views: only the reconstruction half of the concatenated
    # [h, y] output is compared to the input (Eq. 5) -- the embedding half is
    # ignored here, exactly as DEKM_dense.py::loss_train_base does.
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
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
    e_vals, e_vecs = np.linalg.eigh(X)  # Eigenvector v[:, i] corresponds to eigenvalue w[i]; each column is one eigenvector.
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def train(x, y=None, orig_idx=None, random_seed=None):
    # y is only available for labeled benchmark datasets (REUTERS/20NEWS/RCV1);
    # air pollution has no ground truth, so it stays None and silhouette is
    # used instead of ACC/NMI. orig_idx maps each (shuffled) row of x back to
    # its row number in the original, unshuffled source file -- needed so the
    # saved fused embedding/cluster assignment can later be joined back to the
    # original data for re-clustering or interpretation (e.g. land-use/traffic
    # correlation, as in the paper's Section 5.5).
    log_str = f'iter; metric; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
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
    for ite in range(int(140 * 100)):
        if ite % update_interval == 0:
            h1 = model1(x).numpy()[:, :hidden_units]
            h2 = model2(x).numpy()[:, :hidden_units]
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
            H_vt = np.matmul(H, V)  # n,10
            U_vt = np.matmul(U, V)  # n_clusters,10
            #
            loss = np.round(np.mean(loss_value), 5)
            if y is not None:
                acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))
                metric_str = f'acc, nmi = {acc, nmi}'
            else:
                silhouette = silhouette_score(H, assignment)
                metric_str = f'silhouette = {silhouette}'

            # log
            log_str = f'iter {ite // update_interval}; {metric_str}; loss:' \
                      f'{loss}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            print(log_str)
            log_csv(log_str.split(';'), file_name=ds_name)

        if n_change_assignment <= len(x) * 0.001:
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
            h_pred = (y_pred1[:, :hidden_units] + y_pred2[:, :hidden_units]) / 2
            y_pred_cluster = tf.matmul(h_pred, V)
            loss_value = losses.mse(y_true, y_pred_cluster)
        trainable_variables = model1.trainable_variables + model2.trainable_variables
        grads = tape.gradient(loss_value, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    if y is None and orig_idx is not None:
        # Save the final fused embedding + cluster assignment so clustering
        # can be redone (e.g. with a different k) or analyzed (e.g. joined
        # back to the original CSV via orig_index) without retraining.
        if not os.path.exists('output'):
            os.makedirs('output')
        result = pd.DataFrame(H, columns=[f'h_{i}' for i in range(hidden_units)])
        result.insert(0, 'orig_index', orig_idx)
        result['cluster'] = assignment
        result.to_csv(f'output/{ds_name}_clusters.csv', index=False)

    if y is not None:
        return acc, nmi
    return silhouette


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='select dataset:AIRPOLLUTION,REUTERS,20NEWS,RCV1')
    parser.add_argument('ds_name', default='AIRPOLLUTION')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.ds_name is None or args.ds_name not in ['AIRPOLLUTION', 'REUTERS', '20NEWS', 'RCV1']:
        ds_name = 'AIRPOLLUTION'
    else:
        ds_name = args.ds_name

    if ds_name == 'AIRPOLLUTION':
        input_shape = 13
        n_clusters = 4
        hidden_units = 10
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

    run_metrics = []
    time_all_start = time.time()
    for run_index in range(args.runs):
        run_seed = None if args.seed is None else args.seed + run_index
        set_random_seed(run_seed)
        time_start = time.time()
        orig_idx = None
        if ds_name == 'AIRPOLLUTION':
            x, orig_idx = get_x_airpollution(shuffle_seed=run_seed)
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
        metric = train(x, y=y, orig_idx=orig_idx, random_seed=run_seed)
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

    if ds_name == 'AIRPOLLUTION':
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
