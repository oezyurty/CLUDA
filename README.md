# CLUDA: Contrastive Learning for Unsupervised Domain Adaptation of Time Series

### Loading and Preparing Benchmark Datasets
First, create a folder and download the pre-processed versions of the datasets [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B), [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ), and [HHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO). 

In the same folder, you need to run [train_val_test_benchmark.ipynb](train_val_test_benchmark.ipynb) to create valid train-val-test splits (the downloaded datasets initially have only train-test splits). Further, this script will convert the datasets into desired format. This script will create the new versions of the datasets in *./processed_datasets* folder (path relative to the folder in which you initially downloaded the datasets).

### Accessing ICU Datasets MIMIC-IV and AmsterdamUMCdb
If you want to do experiments on MIMIC-IV and AmsterdamUMCdb, first you need to get permission for these datasets even though it is publicly available. 

MIMIC-IV: Access details can be found [here](https://physionet.org/content/mimiciv/0.4/). 

AmsterdamUCMdb: Access details can be found [here](https://amsterdammedicaldatascience.nl).

### Loading ICU Datasets

Once you have access the ICU dataset(s), you can load them with R. For this, you can follow the steps at [load_ICU_datasets](./load_ICU_datasets).

### Preparing ICU Datasets

This step converts the output from previous step into np.array format to be used in Python. For this, you can follow the steps at [prepare_ICU_datasets](./prepare_ICU_datasets).

### Baseline 

We compare our algorithm against the following baselines: [VRADA](https://openreview.net/pdf?id=rk9eAFcxg), [CoDATS](https://dl.acm.org/doi/pdf/10.1145/3394486.3403228), [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf), [CAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf), [CDAN](https://proceedings.neurips.cc/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf), [DDC](https://arxiv.org/pdf/1412.3474.pdf), [DeepCORAL](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35), [DSAN](https://ieeexplore.ieee.org/document/9085896), [HoMM](https://ojs.aaai.org/index.php/AAAI/article/view/5745), and [MMDA](https://arxiv.org/pdf/1901.00282.pdf). 

The implementation of the baselines is adopted from the original papers, and from [AdaTime](https://github.com/emadeldeen24/AdaTime) benchmark suite for time series domain adaptation. 

### CLUDA

We used Python 3.6.5 in our experiments. 

You can install the requirement libraries via `pip install -r requirements.txt` into your new virtual Python environment.

Our main model architecture can be found [here](model/model.py). 

In [main](main), you can find all the scripts to train and evaluate our model. Further, you can find the script for getting the embeddings for your downstream tasks.

For instance, you can train our model with the default arguments as (assuming that you prepared the datasets as in earlier step.)

`python train.py --algo_name cluda --task decompensation --path_src ../Data/miiv_fullstays --path_trg ../Data/aumc_fullstays --experiments_main_folder UDA_decompensation --experiment_folder default` 

All the helper functions/classes are in [utils](utils). If you want to do some changes in dataset format, you can find it useful to check our [ICUDataset](utils/dataset.py) class or [SensorDataset](utils/dataset.py) class.
