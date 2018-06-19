# code

### code/modelling/01_train
Retrieves a model defined in `create_and_compile.py` and creates a model stored on Azure Blob Storage that can later be operationalized. 

Note: operationalization is currently not included in this repository.

### code/modelling/02_evaluate
Jupyter notebook that pulls a separate image dataset and evaluates expected results with those predicted by the model built by `01_train.py`. This is useful in understanding the accuracy of a multi-label classifier.

### code/utilities/keras_hlprs
A group of functions that helps simplify working with Keras and data manipulations. Some of the provided functionality includes
* One Hot Encodings of a Multi-Label Dataframe
* Encoding images as a Numpy Array
* Keras Data Generator for extracting batches from Azure Blob Storage

### code/utilities/rw_hlprs
A group of functions that assist in 
* reading and writing i.r.t Azure Blob Storage
* pickling Keras models for storage and iterable training/evaluating.

Python Styling: [yapf](https://github.com/google/yapf)
