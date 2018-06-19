import os

import numpy as np
import pandas as pd
from azureml.logging import get_azureml_logger
from sklearn.preprocessing import MultiLabelBinarizer

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

logger = get_azureml_logger()


# create a callback to gather azureml logs used for displaying runtime metrics
class AzureMlLoggerCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        logger.log('loss', float(logs.get('loss')))
        logger.log('acc', float(logs.get('acc')))
        logger.log('val_loss', float(logs.get('val_loss')))
        logger.log('val_acc', float(logs.get('val_acc')))


# One-hot Encode Labels
def one_hot_encode(df):
    one_hot = MultiLabelBinarizer()
    df['labels'] = one_hot.fit_transform(df['tags'].str.split(' ')).tolist()
    encoding_len = len(df['labels'][0])

    return df, encoding_len, one_hot


def img_as_array(img, runtime_params={}):
    img.thumbnail(runtime_params['img_resize'])

    # Convert to RGB and normalize
    img_array = np.asarray(img.convert("RGB"), dtype=np.float32)

    img_array = img_array[:, :, ::-1]
    # Zero-center by mean pixel
    img_array[:, :, 0] -= 103.939
    img_array[:, :, 1] -= 116.779
    img_array[:, :, 2] -= 123.68

    return img_array


def get_ImageDataGenerator():
    # Image Augmentation
    # TODO: come back to this and analyze
    return ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)  # randomly flip images horizontally


# def in_mem_gen():
#     datagen = get_ImageDataGenerator()

#     # compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied)
#     datagen.fit(x_train)

#     # fits the model on batches with real-time data augmentation:
#     model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
#                         steps_per_epoch=len(x_train) / 32, epochs=epochs)


def get_data_generator(df, runtime_params, encoding_len, blob_service):
    """
        Returns a batch generator which transforms chunk of raw images into numpy matrices
        and then "yield" them for the classifier. Doing so allow to greatly optimize
        memory usage as the images are processed then deleted by chunks (defined by batch_size)
        instead of preprocessing them all at once and feeding them to the classifier.
        
        Credit: https://github.com/EKami/planet-amazon-deforestation
        
        
        :param batch_size: int
            The batch size
        :return: generator
            The batch generator
        """

    datagen = get_ImageDataGenerator()

    loop_range = len(df)
    while True:
        for i in range(loop_range):

            start_offset = runtime_params['batch_size'] * i

            # The last remaining files could be smaller than the batch_size
            range_offset = min(runtime_params['batch_size'],
                               loop_range - start_offset)

            # If we reached the end of the list then we break the loop
            if range_offset <= 0:
                break

            batch_features = np.zeros((range_offset,
                                       *runtime_params['img_resize'], 3))

            # dimensions => (batch size, size of one-hot encoding)
            batch_labels = np.zeros((range_offset, encoding_len))
            for j in range(range_offset):
                # Maybe shuffle the index?
                path = df.iloc[start_offset + j, 1]
                img_file = path

                # place image blob into a local temp store
                if runtime_params['sample_flag'] != 'true':
                    img_file = 'temp/{}'.format(path.split('/')[1])
                    if not os.path.exists(img_file):
                        blob_service.get_blob_to_path(
                            runtime_params['stor_container'],
                            path,
                            img_file,
                            max_connections=5)
                img = Image.open(img_file)
                img_array = img_as_array(img, runtime_params)

                batch_features[j] = img_array
                batch_labels[j] = df.iloc[start_offset + j, 2]

            # Augment the images (using Keras allow us to add randomization/shuffle to augmented images)
            # Here the next batch of the data generator (and only one for this iteration)
            # is taken and returned in the yield statement
            yield next(
                datagen.flow(batch_features, batch_labels, range_offset))
