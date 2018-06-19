 # Getting Started

## How to Run

In the Azure CLI, prepare the environment and submit the job.

[Help link](https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/cli-for-azure-machine-learning)

```
$ az ml experiment prepare -c (docker | remotevm)
$ az ml experiment submit -c (docker | remotevm) 01_train.py 1e-4 150 10 1 train-jpg/ true
```

* 1e-4 -> learning rate
* 150 -> image size
* 10 -> batch size
* 1 -> epoch
* train-jpg/ -> training data path
* true -> use prior weights

[How to Use Jupyter Notebooks](https://docs.microsoft.com/en-us/azure/machine-learning/preview/how-to-use-jupyter-notebooks)

Go to **Notebooks** tab and select the sample Notebook to view it. To edit and run cells, click **Launch Notebook Server**. Select a kernel, and wait for it to start to run cells.

**_Note_**: Do not select _Python 3_ kernel. Instead, select one of the kernels named after your project. 

You can also launch Notebook server by opening **File**, **Open Command Prompt**, and entering `az ml notebook start`.

## Some Leftover Documentation Covering TDSP

### Team Data Science Process From Microsoft (TDSP)

This repository contains an instantiation of the [**Team Data Science Process (TDSP) from Microsoft**](https://github.com/Azure/Microsoft-TDSP) for project **Azure Machine Learning**. The TDSP is an agile, iterative, data science methodology designed to improve team collaboration and learning. It facilitates better coordinated and more productive data science enterprises by providing:

- a [lifecycle](https://github.com/Azure/Microsoft-TDSP/blob/master/Docs/lifecycle-detail.md) that defines the steps in project development 
- a [standard project structure](https://github.com/Azure/Azure-TDSP-ProjectTemplate)
- artifact templates for reporting
- tools to assist with data science tasks and project execution

### Information About TDSP In Azure Machine Learning
When you instantiate the TDSP from Azure Machine Learning, you get the TDSP-recommended standardized directory structure and document templates for project execution and delivery. The workflow then consists of the following steps:

- modify the documentation templates provided here for your project
- execute your project (fill in with your project's code, documents, and artifact outputs)
- prepare the Data Science deliverables for your client or customer, including the ProjectReport.md report.

We provide [instructions on how to instantiate and use TDSP in Azure Machine Learning](https://aka.ms/how-to-use-tdsp-in-aml).

### Project Folder Structure
The TDSP project template contains following top-level folders:
1. **code**: Contains code
2. **docs**: Contains necessary documentation about the project
3. **sample_data**: Contains **SAMPLE (small)** data that can be used for early development or testing. Typically, not more than several (5) Mbs. Not for full or large data-sets.
