
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at
from utils.keras_utils import train_model
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import os
import numpy as np


if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = list(range(0, 85))
    MODEL_NAME = 'svm'

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'  # can be lstmfcn or alstmfcn

    for DID in DATASET_ID:
        print(DID)
        # Script setup
        sequence_length = MAX_SEQUENCE_LENGTH_LIST[DID]
        nb_classes = NB_CLASSES_LIST[DID]

        dataset_name = TRAIN_FILES[DID][8:-6]

        basepath = 'layer_features/%s/'
        dataset_path = basepath % (dataset_name)
        model_basepath = basepath % (dataset_name)

        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        log_basepath = 'layer_features/logs_raw_svm_data.csv'

        template = '%d,%s,%0.6f\n'

        if not os.path.exists(log_basepath):
            f = open(log_basepath, 'w')
            f.write('id,dataset_name,test_accuracy\n')
            f.flush()
            f.close()

        accuracies = []
        X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DID,
                                                                          normalize_timeseries=True)

        # received as float32, cast to ints
        y_train = (y_train).flatten().round().astype(int)
        y_test = (y_test).flatten().round().astype(int)

        X_train = X_train[:, 0, :]
        X_test = X_test[:, 0, :]

        log_file = open(log_basepath, 'a+')

        CLASSIFIER_PARMAS = [
            {'C': 10.0, 'gamma': 'auto'},
            {'C': 10.0, 'gamma': 'auto'},
            {'C': 10.0, 'gamma': 'auto'},
            {'C': 10.0, 'gamma': 'auto'},
        ]

        kernel = 'linear'

        params = CLASSIFIER_PARMAS[0]

        clf = SVC(kernel=kernel, random_state=0, **params)

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        print()
        print("-" * 50)
        print("Accuracy score : ", accuracy)
        print()

        print()
        print("-" * 50)
        print("Accuracy score : ", accuracy)

        log_file.write(template % (DID, dataset_name, accuracy))
        log_file.flush()
        log_file.close()
