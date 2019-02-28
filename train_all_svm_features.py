
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = list(range(85, 128))  # new datasets
    KERNELS = ['rbf', 'linear']

    MODEL_NAME = 'svm'
    CLASSIFIER_PARMAS = [
        {'C': 10.0, 'gamma': 'auto'},
        {'C': 10.0, 'gamma': 'auto'},
        {'C': 10.0, 'gamma': 'auto'},
        {'C': 10.0, 'gamma': 'auto'},
    ]

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'

    for DID in DATASET_ID:
        # Script setup
        sequence_length = MAX_SEQUENCE_LENGTH_LIST[DID]
        nb_classes = NB_CLASSES_LIST[DID]

        dataset_name = TRAIN_FILES[DID][8:-6]

        basepath = 'lstm_layer_features/%s/'
        dataset_path = basepath % (dataset_name)

        log_basepath = 'lstm_layer_features/logs_svm_update.csv'

        template = '%d,%s,%s,%s,%0.6f\n'

        if not os.path.exists(log_basepath):
            f = open(log_basepath, 'w')
            f.write('id,dataset_name,kernel,feature_type,test_accuracy\n')
            f.flush()
            f.close()

        accuracies = []

        FEATURE_TYPES = ['cnn', 'cnn_mean', 'lstm', 'lstmfcn']

        assert len(CLASSIFIER_PARMAS) == len(FEATURE_TYPES)

        for feature_id, feature_type in enumerate(FEATURE_TYPES):
            for kernel in KERNELS:

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

                model_path = model_basepath + '%s_%s_%s_model.pkl' % (MODEL_NAME, kernel, feature_type)

                params = CLASSIFIER_PARMAS[feature_id]

                clf = SVC(kernel=kernel, random_state=0, **params)

                clf.fit(train_features, train_labels)
                pred = clf.predict(test_features)

                accuracy = accuracy_score(test_labels, pred)

                print()
                print("-" * 50)
                print("Accuracy score : ", accuracy)
                print()

                accuracies.append(accuracy)

                log_file.write(template % (DID, dataset_name, kernel, feature_type, accuracy))
                log_file.flush()
                log_file.close()

                joblib.dump(clf, model_path)

            print()
            print("-" * 50)
            print("SVM on LSTM FCN features : ")
            for feature_type, accuracy in zip(FEATURE_TYPES, accuracies):
                print("Feature type = %s | Test accuracy = %0.6f" % (feature_type, accuracy))

            print()
