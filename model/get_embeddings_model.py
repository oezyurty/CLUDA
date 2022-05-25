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
from utils.augmentations import Augmenter, concat_mask

from model import CLUDA

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

from argparse import ArgumentParser
from collections import namedtuple

def main(args):


    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)
        
    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    #configure our logger
    log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, "embeddings_"+saved_args.log))

    #LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)

    dataset_test_src = ICUDataset(saved_args.path_src, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    dataset_test_trg = ICUDataset(saved_args.path_trg, task=saved_args.task, split_type="test", is_full_subset=True, is_cuda=True)

    augmenter = Augmenter()

    #Calculate the output dim based on the task
    output_dim = 1 if saved_args.task != "los" else 10

    model = CLUDA(num_inputs=(1+saved_args.use_mask)*dataset_test_src[0]['sequence'].shape[1], output_dim = output_dim, num_channels=[64,64,64,64,64], num_static=dataset_test_src[0]['static'].shape[0], 
        use_static=False, mlp_hidden_dim=256, use_batch_norm=saved_args.use_batch_norm, kernel_size=3, dropout=saved_args.dropout, K=saved_args.queue_size,  m=saved_args.momentum)


    model.cuda()

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=batch_size,
                        shuffle=False, num_workers=0)


    model.eval()

    list_embeddings_test_trg = []
    list_y_test_trg = []

    for i_batch, sample_batched in enumerate(dataloader_test_trg):
        
        seq = concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], saved_args.use_mask)

        embeddings = model.get_encoding(seq, is_target=True)

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
        
        seq = concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], saved_args.use_mask)
        
        embeddings = model.get_encoding(seq, is_target=False)

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

    embeddings_save_path = os.path.join(args.experiments_main_folder, args.experiment_folder, "embeddings_test.npz")

    np.savez(embeddings_save_path, embeddings_trg=embeddings_test_trg, y_trg=y_test_trg, embeddings_src=embeddings_test_src, y_src=y_test_src)

    log("All embeddings are saved to " + embeddings_save_path)



# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='DA_experiments')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')


    args = parser.parse_args()

    main(args)
