import os
import pickle
import re
import shutil
import tempfile
import types
from datetime import datetime

from azureml.logging import get_azureml_logger

import keras.models

logger = get_azureml_logger()


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def create_temp_dir():
    if not os.path.exists('temp'):
        os.makedirs('temp')
    else:
        shutil.rmtree('temp')
        os.makedirs('temp')


def load_model(runtime_params, blob_service):
    make_keras_picklable()
    pkl_bytes = blob_service.get_blob_to_bytes(
        runtime_params['stor_container'], 'compiled_model.pkl')
    return pickle.loads(pkl_bytes.content)


def export_trained_model(model, runtime_params, blob_service):
    make_keras_picklable()
    with open('./outputs/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    blob_service.create_blob_from_path(runtime_params['stor_container'],
                                       'trained_model.pkl',
                                       './outputs/trained_model.pkl')


def export_model_weights(model, path, runtime_params, blob_service):
    model.save_weights(path)
    blob_service.create_blob_from_path(
        runtime_params['stor_container'], 'weights-{}.hdf5'.format(
            datetime.utcnow().strftime('%Y%m%d%H%M%S')), path)


def import_weights(runtime_params, blob_service):
    generator = blob_service.list_blobs(runtime_params['stor_container'])
    files = {}
    for blob in generator:
        name = blob.name
        prefix = 'weights'
        if name.startswith(prefix):
            files[re.findall(r"[\w']+", name)[1]] = name

    max_key = max(files, key=int)
    recent_weight = files[str(max_key)]
    print('Loading weight file\t {}'.format(recent_weight))
    logger.log('weights_file', recent_weight)
    blob_service.get_blob_to_path(runtime_params['stor_container'],
                                  recent_weight,
                                  'temp/{}'.format(recent_weight))
    return os.path.join('temp', recent_weight)


def load_trained_model(runtime_params, blob_service):
    make_keras_picklable()
    pkl_bytes = blob_service.get_blob_to_bytes(
        runtime_params['stor_container'], runtime_params['model_pkl_file'])
    return pickle.loads(pkl_bytes.content)


def retrieve_paths(runtime_params, blob_service):
    generator = blob_service.list_blobs(runtime_params['stor_container'])
    paths = []
    for blob in generator:
        if blob.name.startswith(runtime_params['data_path']):
            paths += [blob.name]
        else:
            pass

    return paths
