# CL4UDATS: Contrastive Learning for Unsupervised Domain Adaptation of Time Series

### Accessing ICU Datasets MIMIC-IV and AmsterdamUMCdb
If you want to do experiments on MIMIC-IV and AmsterdamUMCdb, first you need to get permission for these datasets even though it is publicly available. 

MIMIC-IV: Access details can be found [here](https://physionet.org/content/mimiciv/0.4/). 

AmsterdamUCMdb: Access details can be found [here](https://amsterdammedicaldatascience.nl).

### Loading ICU Datasets

Once you have access the ICU dataset(s), you can load them with R. For this, you can follow the steps at [load_ICU_datasets](./load_ICU_datasets).

### Preparing ICU Datasets

This step converts the output from previous step into np.array format to be used in Python. For this, you can follow the steps at [prepare_ICU_datasets](./prepare_ICU_datasets).

### CL4UDATS

Our main model architecture can be found [here](model/model.py). 

In [model](model), you can find all the scripts to train and evaluate our model. Further, you can find the script for getting the embeddings for your downstream tasks.

All the helper functions/classes are in [utils](utils). If you want to do some changes in dataset format, you can find it useful to check our [ICUDataset](utils/dataset.py) class.
