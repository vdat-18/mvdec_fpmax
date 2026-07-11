from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Air pollution dataset (data/preprocessed_data/data_demvk.csv): 13 features,
# n_clusters=4 confirmed via the paper's own Elbow analysis; hidden_units=10
# matches model_view1's embedding width in Fig. 2 (both views must be equal
# width so Eq. 4's averaging fusion works). Only used to fully specify the
# two architectures below -- no training/data-loading wired up yet.
ds_name = 'AIRPOLLUTION'
input_shape = 13
hidden_units = 10


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
