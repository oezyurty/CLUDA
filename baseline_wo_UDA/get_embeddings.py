import sys
sys.path.append("..")
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import ICUDataset
from utils.augmentations import Augmenter
from utils.mlp import MLP
from utils.tcn_classifier import TCN_Classifier

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util_progress_log import AverageMeter, ProgressMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter
from utils.loss import PredictionLoss

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import time
import shutil

import pickle
import json
import logging

from argparse import ArgumentParser
from collections import namedtuple

def main(args):


    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)
        
    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    #configure our logger
    log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, "embeddings_"+saved_args.log))

    #Manually enter the loss weights for the task (for classification tasks)
    #It is to give more weight to minority class
    #loss_weights = (0.1, 0.9)

    weight_ratio = saved_args.weight_ratio

    prop = 1 - (1/(weight_ratio + 1))
                
    weight_0 = 1 / (2 * prop)
    weight_1 = weight_0 * prop / (1 - prop)
    loss_weights = (weight_0, weight_1)
    log("Loss is weighted with " + str(loss_weights))

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

    #LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)

    dataset_test_src = ICUDataset(saved_args.path_src, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    dataset_test_trg = ICUDataset(saved_args.path_trg, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    augmenter = Augmenter()

    #Calculate the output dim based on the task
    output_dim = 1 if saved_args.task != "los" else 10

    feature_extractor = TCN_Classifier(num_inputs=dataset_test_src[0]['sequence'].shape[1], output_dim=output_dim, num_channels=[64,64,64,64,64], num_static=dataset_test_src[0]['static'].shape[0], mlp_hidden_dim=256, use_batch_norm=saved_args.use_batch_norm, use_mask=saved_args.use_mask, kernel_size=3, dropout=saved_args.dropout)


    feature_extractor.cuda()

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
    feature_extractor.load_state_dict(checkpoint['state_dict'])

    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    feature_extractor.eval()

    list_embeddings_test_trg = []
    list_y_test_trg = []

    for i_batch, sample_batched in enumerate(dataloader_test_trg):

        trg_feat = feature_extractor.get_encoding(sample_batched['sequence'])
        
        embeddings = trg_feat.detach().cpu().numpy()
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

        src_feat = feature_extractor.get_encoding(sample_batched['sequence'])

        embeddings = src_feat.detach().cpu().numpy()
        list_embeddings_test_src = list_embeddings_test_src + list(embeddings)

        y_test = sample_batched['label'].detach().cpu().numpy().flatten()

        list_y_test_src = list_y_test_src + list(y_test)

    embeddings_test_src = np.array(list_embeddings_test_src)
    y_test_src = np.array(list_y_test_src)


    log("embeddings are calculated")

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    embeddings_save_path = os.path.join(args.experiments_main_folder, args.experiment_folder, "embeddings_test.npz")

    np.savez(embeddings_save_path, embeddings_trg=embeddings_test_trg, y_trg=y_test_trg, embeddings_src=embeddings_test_src, y_src=y_test_src)

    log("All embeddings are saved to " + embeddings_save_path)
    


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')


    args = parser.parse_args()

    main(args)