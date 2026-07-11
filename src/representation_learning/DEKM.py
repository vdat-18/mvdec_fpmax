import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Model
from utils import get_ACC_NMI
from utils import get_xy
from utils import log_csv
import time
import argparse


def set_random_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def model_conv(load_weights=True):
    init = 'glorot_uniform'
    filters = [32, 64, 128, hidden_units]
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters[0], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        input)
    x = layers.Conv2D(filters[1], kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=init)(
        x)
    x = layers.Conv2D(filters[2], kernel_size=3, strides=2, padding=pad3, activation='relu', kernel_initializer=init)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=filters[-1], name='embed', kernel_initializer=init)(x)
#     x = tf.divide(x, tf.expand_dims(tf.norm(x, 2, -1), -1))
    h = x
    x = layers.Dense(
        filters[2] * (input_shape[0] // 8) * (input_shape[0] // 8),
        activation='relu',
        kernel_initializer=init,
    )(x)
    x = layers.Reshape((input_shape[0] // 8, input_shape[0] // 8, filters[2]))(x)
    x = layers.Conv2DTranspose(
        filters[1], kernel_size=3, strides=2, padding=pad3,
        activation='relu', kernel_initializer=init,
    )(x)
    x = layers.Conv2DTranspose(
        filters[0], kernel_size=5, strides=2, padding='same',
        activation='relu', kernel_initializer=init,
    )(x)
    x = layers.Conv2DTranspose(
        input_shape[2], kernel_size=5, strides=2, padding='same',
        kernel_initializer=init,
    )(x)
    output = layers.Concatenate()([h,
                                   layers.Flatten()(x)])
    model = Model(inputs=input, outputs=output)
    # model.summary()
    if load_weights:
        model.load_weights(f'weight_base_{ds_name}.weights.h5')
        print('model_conv: weights was loaded')
    return model


def loss_train_base(y_true, y_pred):
    y_true = layers.Flatten()(y_true)
    y_pred = y_pred[:, hidden_units:]
    return losses.mse(y_true, y_pred)


def train_base(ds_xx):
    model = model_conv(load_weights=False)
    model.compile(optimizer='adam', loss=loss_train_base)
    model.fit(ds_xx, epochs=pretrain_epochs, verbose=2)
    model.save_weights(f'weight_base_{ds_name}.weights.h5')


def sorted_eig(X):
    X = (X + X.T) / 2
    e_vals, e_vecs = np.linalg.eigh(X)  # Eigenvector v[:, i] corresponds to eigenvalue w[i]; each column is one eigenvector.
    idx = np.argsort(e_vals)
    e_vecs = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return e_vals, e_vecs


def train(x, y, random_seed=None):
    log_str = f'iter; acc, nmi, ri ; loss; n_changed_assignment; time:{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
    log_csv(log_str.split(';'), file_name=ds_name)
    model = model_conv()

    optimizer = tf.keras.optimizers.Adam()
    loss_value = 0
    acc = np.nan
    nmi = np.nan
    index = 0
    kmeans_n_init = 100
    assignment = np.array([-1] * len(x))
    index_array = np.arange(x.shape[0])
    for ite in range(int(140 * 100)):
        if ite % update_interval == 0:
            H = model(x).numpy()[:, :hidden_units]
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
            H_vt = np.matmul(H, V)  # 1000,5
            U_vt = np.matmul(U, V)  # 10,5
            #
            loss = np.round(np.mean(loss_value), 5)
            acc, nmi = get_ACC_NMI(np.array(y), np.array(assignment))

            # log
            log_str = f'iter {ite // update_interval}; acc, nmi, ri = {acc, nmi, loss}; loss:' \
                      f'{loss:.5f}; n_changed_assignment:{n_change_assignment}; time:{time.time() - time_start:.3f}'
            print(log_str)
            log_csv(log_str.split(';'), file_name=ds_name)

        if n_change_assignment <= len(x) * 0.001:
            model.save_weights(f'weight_final_l2_{ds_name}.weights.h5')
            print('end')
            break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
        y_true = H_vt[idx]
        temp = assignment[idx]
        for i in range(len(idx)):
            y_true[i, -1] = U_vt[temp[i], -1]

        with tf.GradientTape() as tape:
            # tape.watch(model.trainable_variables)
            y_pred = model(x[idx])
            y_pred_cluster = tf.matmul(y_pred[:, :hidden_units], V)
            loss_value = losses.mse(y_true, y_pred_cluster)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    return acc, nmi


if __name__ == '__main__':
    pretrain_epochs = 200
    pretrain_batch_size = 256
    batch_size = 256
    update_interval = 40
    parser = argparse.ArgumentParser(description='select dataset:MNIST,COIL20,FRGC,USPS')
    parser.add_argument('ds_name', default='MNIST')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.ds_name is None or not args.ds_name in ['MNIST', 'FRGC', 'COIL20', 'USPS']:
        ds_name = 'MNIST'
    else:
        ds_name = args.ds_name
        
    if ds_name == 'MNIST':
        input_shape = (28, 28, 1)
        n_clusters = 10
    elif ds_name == 'USPS':
        input_shape = (16, 16, 1)
        n_clusters = 10
    elif ds_name == 'COIL20':
        input_shape = (28, 28, 1)
        n_clusters = 20
    elif ds_name == 'FRGC':
        input_shape = (32, 32, 3)
        n_clusters = 20

    hidden_units = n_clusters

    run_metrics = []
    time_all_start = time.time()
    for run_index in range(args.runs):
        run_seed = None if args.seed is None else args.seed + run_index
        set_random_seed(run_seed)
        time_start = time.time()
        x, y = get_xy(ds_name=ds_name, shuffle_seed=run_seed)
        ds_xx = tf.data.Dataset.from_tensor_slices((x, x)).shuffle(
            8000, seed=run_seed
        ).batch(pretrain_batch_size)
        train_base(ds_xx)
        acc, nmi = train(x, y, random_seed=run_seed)
        run_metrics.append((acc, nmi))
        run_str = (
            f'run {run_index + 1}/{args.runs}; seed:{run_seed}; '
            f'acc:{acc}; nmi:{nmi}; time:{time.time() - time_start:.3f}'
        )
        print(run_str)
        log_csv(run_str.split(';'), file_name=ds_name)

    metrics = np.asarray(run_metrics, dtype=float)
    avg_acc = float(np.nanmean(metrics[:, 0]))
    avg_nmi = float(np.nanmean(metrics[:, 1]))
    avg_str = (
        f'average over {args.runs} runs; acc:{avg_acc:.5f}; '
        f'nmi:{avg_nmi:.5f}; time:{time.time() - time_all_start:.3f}'
    )
    print(avg_str)
    log_csv(avg_str.split(';'), file_name=ds_name)

