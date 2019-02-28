
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at
from utils.keras_utils import train_model
from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K

import os
import numpy as np
import joblib


if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = list(range(0, 85))
    MODEL_NAME = 'mlp'

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'  # can be lstmfcn or alstmfcn

    for DID in DATASET_ID:
        # Script setup
        sequence_length = MAX_SEQUENCE_LENGTH_LIST[DID]
        nb_classes = NB_CLASSES_LIST[DID]

        dataset_name = TRAIN_FILES[DID][8:-6]

        # change basepath from lstm or asltm
        basepath = 'lstm_layer_features/%s/'
        dataset_path = basepath % (dataset_name)
        model_basepath = basepath % (dataset_name)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        # change log_basepath from lstm or asltm
        log_basepath = 'lstm_layer_features/logs_raw_mlp_data.csv'

        template = '%d,%s,%0.6f\n'

        if not os.path.exists(log_basepath):
            f = open(log_basepath, 'w')
            f.write('id,dataset_name,test_accuracy\n')
            f.flush()
            f.close()

        accuracies = []
        X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DID, normalize_timeseries=True)

        X_train = X_train[:, 0, :]
        X_test = X_test[:, 0, :]

        print("Train shape : ", X_train.shape, y_train.shape)
        print("Test shape : ", X_test.shape, y_test.shape)
        print()

        model_path = model_basepath + '%s_%s_model.h5' % (MODEL_NAME, 'raw_data')
        log_file = open(log_basepath, 'a+')

        num_classes = len(np.unique(y_train))

        classes = np.unique(y_train)
        le = LabelEncoder()
        y_ind = le.fit_transform(y_train.ravel())
        recip_freq = len(y_train) / (len(le.classes_) *
                                     np.bincount(y_ind).astype(np.float64))
        class_weight = recip_freq[le.transform(classes)]

        print("Class weights : ", class_weight)

        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        K.clear_session()

        ip = Input(shape=(X_train.shape[-1],))
        out = Dense(num_classes, activation='softmax',
                    kernel_initializer='ones')(ip)

        model = Model(ip, out)

        optimizer = Adam(0.001)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        factor = 1. / np.cbrt(2)

        checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1,
                                     save_weights_only=True, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                      factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
        callbacks = [checkpoint, reduce_lr]

        model.fit(X_train, y_train_cat, batch_size=128, epochs=300, class_weight=class_weight,
                  callbacks=callbacks, validation_data=(X_test, y_test_cat))

        scores = model.evaluate(X_test, y_test_cat, batch_size=128)
        accuracy = scores[-1]

        print()
        print("-" * 50)
        print("Accuracy score : ", accuracy)

        log_file.write(template % (DID, dataset_name, accuracy))
        log_file.flush()
        log_file.close()
