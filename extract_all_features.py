from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, CuDNNLSTM, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at
from utils.keras_utils import extract_features
from utils.layer_utils import AttentionLSTM

import os
import numpy as np
from keras import backend as K


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_attention_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = list(range(85, 128))  # All new datasets
    num_cells = 8
    model_fn = generate_lstmfcn  # Select model to build

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'

    for DID in DATASET_ID:
        K.clear_session()

        # Script setup
        sequence_length = MAX_SEQUENCE_LENGTH_LIST[DID]
        nb_classes = NB_CLASSES_LIST[DID]
        model = model_fn(sequence_length, nb_classes, num_cells)

        base_weights_dir = '%s_%d_cells_weights/'
        dataset_name = TRAIN_FILES[DID][8:-6]
        weights_dir = base_weights_dir % (model_name, num_cells)

        dataset_name = weights_dir + dataset_name

        extract_features(model, DID, dataset_name, layer_name='cnn',
                         normalize_timeseries=True)

        extract_features(model, DID, dataset_name, layer_name='lstm',
                         normalize_timeseries=True)

        extract_features(model, DID, dataset_name, layer_name='lstmfcn',
                         normalize_timeseries=True)
