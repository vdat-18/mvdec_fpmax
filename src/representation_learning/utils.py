import csv
import os

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist, mnist

from pipeline.external_metrics import compute_external_metrics


def get_ACC_NMI(_y, _y_pred):
    metrics = compute_external_metrics(
        _y,
        _y_pred,
        n_clusters=len(np.unique(_y)),
    )
    return np.round(metrics.acc, 5), np.round(metrics.nmi, 5)


def get_xy(ds_name='REUTERS', dir_path=r'datasets/', log_print=True, shuffle_seed=None):
    dir_path = dir_path + ds_name + '/'
    if ds_name == 'REUTERS':
        x = np.load(dir_path + '10k_feature.npy').astype(np.float32)
        y = np.load(dir_path + '10k_target.npy').astype(np.int32)
        x = x / tf.expand_dims(tf.norm(x, ord=2, axis=-1), -1).numpy()
    elif ds_name == '20NEWS':
        local_files = [
            'train_data.npz', 'train_label.npz',
            'test_data.npz', 'test_label.npz',
        ]
        if all(os.path.exists(dir_path + file_name) for file_name in local_files):
            import scipy.sparse as sp

            def load_npz_array(file_name):
                try:
                    return sp.load_npz(dir_path + file_name).toarray()
                except Exception:
                    with np.load(dir_path + file_name, allow_pickle=True) as data:
                        return data[data.files[0]]

            x = np.concatenate([
                load_npz_array('train_data.npz'),
                load_npz_array('test_data.npz'),
            ]).astype(np.float32)
            y = np.concatenate([
                load_npz_array('train_label.npz'),
                load_npz_array('test_label.npz'),
            ]).reshape(-1).astype(np.int32)
            print("Dataset 20NEWS loaded from local files...")
        else:
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import TfidfVectorizer
            # Fetch the dataset
            _20news = fetch_20newsgroups(subset="all")
            print("Dataset 20NEWS loaded from sklearn...")
            data = _20news.data
            y = _20news.target
            vectorizer = TfidfVectorizer(max_features=2000)
            data = vectorizer.fit_transform(data)  # Keep only the 2000 top words in the vocabulary
            x = data.toarray().astype(np.float32)  # Switch from sparse matrix to full matrix
        norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        norm[norm == 0] = 1
        x = x / norm
    elif ds_name == 'RCV1':
        import scipy.sparse as sp
        from sklearn.datasets import fetch_rcv1

        # Fetch the dataset
        dataset = fetch_rcv1(subset="all")
        print("Dataset RCV1 loaded...")
        data = dataset.data
        target = dataset.target

        # Get the split between training/test set and validation set
        test_indices = read_list(dir_path + "test")
        n_test = test_indices.shape[0]
        validation_indices = read_list(dir_path + "validation")
        n_validation = validation_indices.shape[0]

        # Filter the dataset
        ## Keep only the data points in the test and validation sets
        test_data = data[test_indices]
        test_target = target[test_indices]
        validation_data = data[validation_indices]
        validation_target = target[validation_indices]
        data = sp.vstack([test_data, validation_data])
        target = sp.vstack([test_target, validation_target])
        ## Update test_indices and validation_indices to fit the new data indexing
        test_indices = np.asarray(range(0, n_test))  # Test points come first in filtered dataset
        validation_indices = np.asarray(
            range(n_test, n_test + n_validation))  # Validation points come after in filtered dataset

        # Pre-process the dataset
        ## Filter words based on tf-idf
        sum_tfidf = np.asarray(data.sum(axis=0))[
            0]  # Sum of tf-idf for all words based on the filtered dataset
        word_indices = np.argpartition(-sum_tfidf, 2000)[:2000]  # Keep only the 2000 top words in the vocabulary
        data = data[:, word_indices].toarray()  # Switch from sparse matrix to full matrix
        ## Retrieve the unique label (corresponding to one of the specified categories) from target's label vector
        names = dataset.target_names
        category_names = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
        category_indices = [i for i in range(len(names)) if names[i] in category_names]
        dict_category_indices = {j: i for i, j in
                                 enumerate(category_indices)}  # To rescale the indices between 0 and some K
        filtered_target = []
        for i in range(target.shape[0]):  # Loop over data points
            target_coo = target[i].tocoo().col
            filtered_target_coo = [t for t in target_coo if t in category_indices]
            assert len(filtered_target_coo) == 1  # Only one relevant label per document because of pre-filtering
            filtered_target.append(dict_category_indices[filtered_target_coo[0]])
        y = np.asarray(filtered_target)
        x = data
        x = x / tf.expand_dims(tf.norm(x, ord=2, axis=-1), -1).numpy()
    elif ds_name == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)
    elif ds_name == 'FASHION':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = np.expand_dims(np.divide(x, 255.), -1)
    elif ds_name == 'USPS':
        with h5py.File(dir_path + 'USPS.h5', 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
        x = np.concatenate([X_tr, X_te], 0)
        y = np.concatenate([y_tr, y_te], 0)
        x = np.reshape(x, (len(x), 16, 16)).astype(np.float32)
    elif ds_name == 'COIL20':
        f = h5py.File(dir_path + 'COIL20.h5', 'r')
        x = np.array(f['data'][()]).squeeze()
        x = np.expand_dims(np.swapaxes(x, 1, 2).astype(np.float32), -1)
        x = tf.image.resize(x, [28, 28]).numpy()
        x = x / 255.
        y = np.array(f['labels'][()]).astype(np.float32)
        y[y == 20.] = 0.
    elif ds_name == 'FRGC':
        with h5py.File(dir_path + 'FRGC.h5', 'r') as hf:
            data = hf.get('data')[:]
            data = np.swapaxes(data, 1, 3)
            x = data / 255.
            y = hf.get('labels')[:]
            y_unique = np.unique(y)
            for i in range(len(y_unique)):
                y[y == y_unique[i]] = i
    if shuffle_seed is None:
        shuffle_seed = int(np.random.randint(100))
    idx = np.arange(0, len(x))
    idx = tf.random.shuffle(idx, seed=shuffle_seed).numpy()
    x = x[idx]
    y = y[idx]
    # x = tf.random.shuffle(x, seed=shuffle_seed).numpy()
    # y = tf.random.shuffle(y, seed=shuffle_seed).numpy()
    if log_print:
        print(ds_name)
    return x, y


def log_csv(strToWrite, file_name):
    path = r'log_history/'
    if not os.path.exists(path):
        os.makedirs(path)
    f = open(path + file_name + '.csv', 'a+', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(strToWrite)
    f.close()


def read_list(file_name, type='int'):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    if type == 'str':
        array = np.asarray([l.strip() for l in lines])
        return array
    elif type == 'int':
        array = np.asarray([int(l.strip()) for l in lines])
        return array
    else:
        print("Unknown type")
        return None
