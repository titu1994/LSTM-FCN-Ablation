
import os

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES

if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = list(range(85, 128))  # new datasets

    MODEL_NAME = 'mlp'

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'  # can be lstmfcn or alstmfcn

    for DID in DATASET_ID:
        # Script setup
        sequence_length = MAX_SEQUENCE_LENGTH_LIST[DID]
        nb_classes = NB_CLASSES_LIST[DID]

        dataset_name = TRAIN_FILES[DID][8:-6]

        basepath = 'lstm_layer_features/%s/'
        dataset_path = basepath % (dataset_name)

        log_basepath = 'lstm_layer_features/mlp_logs_proper.csv'

        template = '%d,%s,%s,%0.6f\n'

        if not os.path.exists(log_basepath):
            f = open(log_basepath, 'w')
            f.write('id,dataset_name,feature_type,test_accuracy\n')
            f.flush()
            f.close()

        accuracies = []

        FEATURE_TYPES = ['cnn_mean', 'lstm']

        for feature_id, feature_type in enumerate(FEATURE_TYPES):
            model_basepath = basepath % (dataset_name)
            log_file = open(log_basepath, 'a+')

            if feature_type == 'cnn_mean':
                average_cnn = True
                feature_type = 'cnn'
            else:
                average_cnn = False

            train_basepath = dataset_path + '%s_%s_train' % (feature_type, dataset_name)
            train_features = np.load(train_basepath + '_features.npy')
            train_labels = np.load(train_basepath + '_labels.npy').flatten().round().astype(int)

            test_basepath = dataset_path + '%s_%s_test' % (feature_type, dataset_name)
            test_features = np.load(test_basepath + '_features.npy')
            test_labels = np.load(test_basepath + '_labels.npy').flatten().round().astype(int)

            if average_cnn:
                shape = train_features.shape
                train_features = train_features.reshape((shape[0], -1, 128))
                train_features = np.mean(train_features, axis=1)

                shape = test_features.shape
                test_features = test_features.reshape((shape[0], -1, 128))
                test_features = np.mean(test_features, axis=1)

            print("Feature type : ", feature_type)
            print("Train shape : ", train_features.shape, train_labels.shape)
            print("Test shape : ", test_features.shape, test_labels.shape)
            print()

            if average_cnn:
                feature_type = 'cnn_mean'

            model_path = model_basepath + '%s_%s_model.h5' % (MODEL_NAME, feature_type)
            num_classes = len(np.unique(train_labels))

            classes = np.unique(train_labels)
            le = LabelEncoder()
            y_ind = le.fit_transform(train_labels.ravel())
            recip_freq = len(train_labels) / (len(le.classes_) *
                                              np.bincount(y_ind).astype(np.float64))
            class_weight = recip_freq[le.transform(classes)]

            print("Class weights : ", class_weight)

            train_labels_cat = to_categorical(train_labels, num_classes)
            test_labels_cat = to_categorical(test_labels, num_classes)

            K.clear_session()

            ip = Input(shape=(train_features.shape[-1],))
            out = Dense(num_classes, activation='softmax',
                        kernel_initializer='ones')(ip)

            model = Model(ip, out)

            optimizer = Adam(0.001)
            model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            factor = 1. / np.cbrt(2)

            checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1,
                                         save_weights_only=True, save_best_only=True)

            reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
                                          factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

            callbacks = [checkpoint, reduce_lr]

            model.fit(train_features, train_labels_cat, batch_size=128, epochs=2000,
                      callbacks=callbacks, validation_data=(test_features, test_labels_cat),
                      class_weight=class_weight, verbose=2)

            scores = model.evaluate(test_features, test_labels_cat, batch_size=128)
            accuracy = scores[-1]

            print()
            print("-" * 50)
            print("Accuracy score : ", accuracy)
            print()

            accuracies.append(accuracy)

            log_file.write(template % (DID, dataset_name, feature_type, accuracy))
            log_file.flush()
            log_file.close()

        print()
        print("-" * 50)
        print("SVM on LSTM FCN features : ")
        for feature_type, accuracy in zip(FEATURE_TYPES, accuracies):
            print("Feature type = %s | Test accuracy = %0.6f" % (feature_type, accuracy))

        print()
