import os
import pickle
import sys
from code.utilities import rw_hlprs

from azure.storage.blob import BlockBlobService
from azureml.logging import get_azureml_logger

import tensorflow as tf
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam

logger = get_azureml_logger()

stor_acct_name = os.environ.get("ACCOUNT_NAME")
stor_acct_key = os.environ.get("ACCOUNT_KEY")
stor_container = os.environ.get("CONTAINER_NAME")
blob_service = BlockBlobService(
    account_name=stor_acct_name, account_key=stor_acct_key)


def create_model(img_resize, learning_rate, encoding_len):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(*img_resize, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # the model so far outputs 3D feature maps (height, width, features)

    model.add(
        Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(encoding_len, activation='sigmoid'))

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


def main():
    # load regularization rate from argument if present
    runtime_params = {}
    if len(sys.argv) > 1:
        resize = int(sys.argv[1])
        runtime_params['img_resize'] = (resize, resize)
        runtime_params['learning_rate'] = float(sys.argv[2])
        runtime_params['encoding_len'] = int(sys.argv[3])

    model = create_model(runtime_params['img_resize'],
                         runtime_params['learning_rate'],
                         runtime_params['encoding_len'])

    # write ops
    with open('./outputs/compiled_model.pkl', 'wb') as f:
        rw_hlprs.make_keras_picklable()
        pickle.dump(model, f)

    blob_service.create_blob_from_path(stor_container, 'compiled_model.pkl',
                                       './outputs/compiled_model.pkl')
    blob_service.create_blob_from_path(stor_container, 'training_data.pkl',
                                       './outputs/training_data.pkl')


if __name__ == '__main__':
    main()
