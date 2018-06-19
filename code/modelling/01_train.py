import os
import shutil
import sys

import pandas as pd
from azure.storage.blob import BlockBlobService
from azureml.logging import get_azureml_logger

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utilities import keras_hlprs, rw_hlprs

from . import create_and_compile

logger = get_azureml_logger()


def create_dataframes(runtime_params, blob_service):
    train_paths = rw_hlprs.retrieve_paths(runtime_params, blob_service)
    blob_service.get_blob_to_path(runtime_params['stor_container'],
                                  'train_v3.csv', 'temp/train_v3.csv')

    label_df = pd.DataFrame.from_csv(
        'temp/train_v3.csv', header=0, index_col='image_name')
    img_path_df = pd.DataFrame.from_records(
        [(img_path.split('/')[1][:-4], img_path) for img_path in train_paths],
        columns=['image_name', 'img_path'])

    # create final training set DataFrame format
    train_img_df = label_df.join(img_path_df.set_index('image_name'))
    train_img_df, encoding_len, _ = keras_hlprs.one_hot_encode(train_img_df)

    # split into .8/.2 training/validation datasets
    val_img_df = train_img_df.sample(frac=.2, random_state=7).dropna()
    train_img_df = train_img_df[~train_img_df.isin(val_img_df)].dropna()

    # Make sure there are no records shared between val and train
    intersect = val_img_df.index.intersection(train_img_df.index)
    assert len(intersect) == 0

    return (train_img_df, val_img_df, encoding_len)


def train(model, epochs, train_df, val_df, encoding_len, weights_path,
          runtime_params, blob_service):
    train_generator = keras_hlprs.get_data_generator(
        train_df, runtime_params, encoding_len, blob_service)
    validation_generator = keras_hlprs.get_data_generator(
        val_df, runtime_params, encoding_len, blob_service)

    steps = len(train_df) / runtime_params['batch_size']

    checkpointer = ModelCheckpoint(
        filepath=weights_path, save_best_only=True, period=3)
    history = keras_hlprs.AzureMlLoggerCallback()
    early_stopping = EarlyStopping(
        monitor='val_acc', min_delta=0.001, patience=3, verbose=0, mode='auto')

    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=steps,
        callbacks=[checkpointer, history, early_stopping])

    # always save your weights after training or during training
    rw_hlprs.export_model_weights(model, weights_path, runtime_params,
                                  blob_service)

    return model


def main():
    runtime_params = {}
    # set runtime parameters
    if len(sys.argv) > 1:
        runtime_params['learning_rate'] = float(sys.argv[1])

        resize = int(sys.argv[2])
        runtime_params['img_resize'] = (resize, resize)

        runtime_params['batch_size'] = int(sys.argv[3])
        runtime_params['epochs'] = int(sys.argv[4])
        runtime_params['data_path'] = str(sys.argv[5])
        runtime_params['use_weights'] = str(sys.argv[6]).lower() == 'true'

        runtime_params['stor_acct_name'] = os.environ.get("ACCOUNT_NAME")
        runtime_params['stor_acct_key'] = os.environ.get("ACCOUNT_KEY")
        runtime_params['stor_container'] = os.environ.get("CONTAINER_NAME")

        logger.log('learning_rate', runtime_params['learning_rate'])
        logger.log('epochs', runtime_params['epochs'])
        logger.log('batch_size', runtime_params['batch_size'])
        logger.log('use_weights', runtime_params['use_weights'])
    else:
        exit

    rw_hlprs.create_temp_dir()

    blob_service = BlockBlobService(
        account_name=runtime_params['stor_acct_name'],
        account_key=runtime_params['stor_acct_key'])

    train_img_df, val_img_df, encoding_len = create_dataframes(
        runtime_params, blob_service)

    model = create_and_compile.create_model(runtime_params['img_resize'],
                                            runtime_params['learning_rate'],
                                            encoding_len)

    # optionally load previous weights
    weights_path = os.path.join('./outputs', 'weights.hdf5')
    if (runtime_params['use_weights']):
        model.load_weights(
            rw_hlprs.import_weights(runtime_params, blob_service))

    model = train(model, runtime_params['epochs'], train_img_df, val_img_df,
                  encoding_len, weights_path, runtime_params, blob_service)

    # cleanup
    rw_hlprs.export_trained_model(model, runtime_params, blob_service)
    shutil.rmtree('./temp')


if __name__ == '__main__':
    main()
