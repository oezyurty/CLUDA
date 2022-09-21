import sys
sys.path.append("..")
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import ICUDataset, SensorDataset, get_dataset
from utils.augmentations import Augmenter
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util_progress_log import AverageMeter, ProgressMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss

from models.models import ReverseLayerF

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import time
import shutil

import pickle
import json
import logging

from argparse import ArgumentParser
from collections import namedtuple

from algorithms import get_algorithm


def main(args):

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)
        
    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    #configure our logger
    log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, "embedding_"+saved_args.log))

    #Manually enter the loss weights for the task (for classification tasks)
    #It is to give more weight to minority class
    #loss_weights = (0.1, 0.9)

    #Some functions and variables for logging 
    dataset_type = get_dataset_type(saved_args)

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

    #LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)

    #dataset_test_src = ICUDataset(saved_args.path_src, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    #dataset_test_trg = ICUDataset(saved_args.path_trg, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test")

    dataset_test_trg = get_dataset(saved_args, domain_type="target", split_type="test")

    augmenter = Augmenter()

    #Calculate input_channels_dim and input_static_dim 
    input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
    input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0

    #Get our algorithm
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    algorithm.load_state(experiment_folder_path)

    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    #turn algorithm into eval mode
    algorithm.eval()

    list_embeddings_test_trg = []
    list_y_test_trg = []

    for i_batch, sample_batched in enumerate(dataloader_test_trg):

        embeddings = algorithm.get_embedding(sample_batched)

        embeddings = embeddings.detach().cpu().numpy()
        list_embeddings_test_trg = list_embeddings_test_trg + list(embeddings)

        y_test = sample_batched['label'].detach().cpu().numpy().flatten()
            
        list_y_test_trg = list_y_test_trg + list(y_test)


    embeddings_test_trg = np.array(list_embeddings_test_trg)
    y_test_trg = np.array(list_y_test_trg)

    log("Target embeddings are calculated")

    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")


    list_embeddings_test_src = []
    list_y_test_src= []

    for i_batch, sample_batched in enumerate(dataloader_test_src):

        embeddings = algorithm.get_embedding(sample_batched)

        embeddings = embeddings.detach().cpu().numpy()
        list_embeddings_test_src = list_embeddings_test_src + list(embeddings)

        y_test = sample_batched['label'].detach().cpu().numpy().flatten()

        list_y_test_src = list_y_test_src + list(y_test)

    embeddings_test_src = np.array(list_embeddings_test_src)
    y_test_src = np.array(list_y_test_src)

    log("embeddings are calculated")

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    save_folder = args.output_folder if args.output_folder!='' else os.path.join(args.experiments_main_folder, args.experiment_folder)
    save_file_name = args.output_name+".npz" if args.output_name !='' else "embeddings_test.npz"

    embeddings_save_path = os.path.join(save_folder, save_file_name)

    np.savez(embeddings_save_path, embeddings_trg=embeddings_test_trg, y_trg=y_test_trg, embeddings_src=embeddings_test_src, y_src=y_test_src)

    log("All embeddings are saved to " + embeddings_save_path)



# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('-of', '--output_folder', type=str, default='')
    parser.add_argument('-on', '--output_name', type=str, default='')


    args = parser.parse_args()

    main(args)