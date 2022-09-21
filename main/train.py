import sys
sys.path.append("..")
import os
import numpy as np
import random
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

    #Adjust the seed for controlling the randomness
    # Torch RNG
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Python RNG
    np.random.seed(args.seed)
    random.seed(args.seed)

    #First configure our logger
    log = get_logger(os.path.join(args.experiments_main_folder, args.experiment_folder, args.log))

    #Create PredictionLoss object for loss calculation of the main task
    #pred_loss = PredictionLoss(args.task, args.weight_ratio)

    #log("Loss is weighted with " + str(pred_loss.loss_weights))

    #print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])

    #log("The device type is " + str(torch.cuda.get_device_properties(0)))

    #Some functions and variables for logging 
    dataset_type = get_dataset_type(args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type == "icu":
            if args.task != "los":
                log("ROC AUC score is : %.4f " % (metrics_pred["roc_auc"]))
                log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            else:
                log("KAPPA score is : %.4f " % (metrics_pred["kappa"]))
        #In case dataset type is sensor
        else:  
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))


    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_val_iteration = args.num_val_iteration

    #LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)
    """
    dataset_src = ICUDataset(args.path_src, task=args.task, split_type="train", is_full_subset=True, is_cuda=True)
    dataset_val_src = ICUDataset(args.path_src, task=args.task, split_type="val", is_full_subset=True, is_cuda=True)
    dataset_test_src = ICUDataset(args.path_src, task=args.task, split_type="test", is_full_subset=True, is_cuda=True)


    dataset_trg = ICUDataset(args.path_trg, task=args.task, split_type="train", is_full_subset=True, is_cuda=True)
    dataset_val_trg = ICUDataset(args.path_trg, task=args.task, split_type="val", is_full_subset=True, is_cuda=True)
    dataset_test_trg = ICUDataset(args.path_trg, task=args.task, split_type="test", is_full_subset=True, is_cuda=True)
    """

    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    dataset_val_src = get_dataset(args, domain_type="source", split_type="val")
    dataset_test_src = get_dataset(args, domain_type="source", split_type="test")

    dataset_trg = get_dataset(args, domain_type="target", split_type="train")
    dataset_val_trg = get_dataset(args, domain_type="target", split_type="val")
    dataset_test_trg = get_dataset(args, domain_type="target", split_type="test")



    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                            shuffle=True, num_workers=0)

    dataloader_trg= DataLoader(dataset_trg, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    dataloader_val_trg= DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                            shuffle=True, num_workers=0)

    #Check that we are not iterating more than the length of dataloaders
    #assert num_val_iteration < len(dataloader_val_src)
    #assert num_val_iteration < len(dataloader_val_trg)
    max_num_val_iteration = min(len(dataloader_val_src), len(dataloader_val_trg))
    if max_num_val_iteration < num_val_iteration:
        num_val_iteration = max_num_val_iteration

    #Calculate input_channels_dim and input_static_dim 
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0

    #Get our algorithm
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)


    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    #Initialize progress metrics before training
    count_step = 0
    best_val_score = -100

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    for i in range(args.num_epochs):

        end = time.time()
        
        dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                            shuffle=True, num_workers=0)
        dataloader_iterator = iter(dataloader_trg)
        
        for i_batch, sample_batched_src in enumerate(dataloader_src):
            #Current model does not support smaller batches than batch_size (due to queue ptr)
            if len(sample_batched_src['sequence']) != batch_size:
                continue
                
            #Since AUMC is a smaller dataset, if we are out of batch, initialize iterator once again.
            try:
                sample_batched_trg = next(dataloader_iterator)
            except StopIteration:
                dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_iterator = iter(dataloader_trg)
                sample_batched_trg = next(dataloader_iterator)
            
            #Current model does not support smaller batches than batch_size (due to queue ptr)
            if len(sample_batched_trg['sequence']) != batch_size:
                continue
            
            # measure data loading time
            data_time.update(time.time() - end)

            #Training step of algorithm
            algorithm.step(sample_batched_src, sample_batched_trg, count_step=count_step)

            count_step+=1

            if count_step >= args.num_steps+1:
                break

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if count_step % args.checkpoint_freq == 0:

                progress = ProgressMeter(
                len(dataloader_src),
                [batch_time, data_time] + algorithm.return_metrics(),
                prefix="Epoch: [{}]".format(i))

                log(progress.display(i_batch+1, is_logged=True))

                #Refresh the saved metrics for algorithm
                algorithm.init_metrics()

                #Refresh the validation meters of algorithm
                algorithm.init_pred_meters_val()

                #turn algorithm into eval mode
                algorithm.eval()

                #Timings are refreshed
                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                end = time.time()

                dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_val_src_iterator = iter(dataloader_val_src)
                
                dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_val_trg_iterator = iter(dataloader_val_trg)
                
                
                for i_batch_val in range(num_val_iteration):
                    sample_batched_val_src = next(dataloader_val_src_iterator)
                    sample_batched_val_trg = next(dataloader_val_trg_iterator)

                    # measure data loading time
                    data_time.update(time.time() - end)

                    #Validation step of algorithm
                    algorithm.step(sample_batched_val_src, sample_batched_val_trg, count_step=count_step)

                progress = ProgressMeter(
                num_val_iteration,
                [batch_time, data_time] + algorithm.return_metrics(),
                prefix="Epoch: [{}]".format(i))

                log("VALIDATION RESULTS")
                log(progress.display(i_batch_val+1, is_logged=True))
                
                log("VALIDATION SOURCE PREDICTIONS")

                metrics_pred_val_src = algorithm.pred_meter_val_src.get_metrics()
                """
                if args.task != "los":
                    log("ROC AUC score is : %.4f " % (metrics_pred_val_src["roc_auc"]))
                    log("AUPRC score is : %.4f " % (metrics_pred_val_src["avg_prc"]))
                else:
                    log("KAPPA score is : %.4f " % (metrics_pred_val_src["kappa"]))
                """
                log_scores(args, dataset_type, metrics_pred_val_src)

                if dataset_type == "icu":
                    cur_val_score = metrics_pred_val_src["avg_prc"] if args.task != "los" else metrics_pred_val_src["kappa"]
                else:
                    cur_val_score = metrics_pred_val_src["mac_f1"]

                if cur_val_score > best_val_score:
                    algorithm.save_state(experiment_folder_path)

                    best_val_score = cur_val_score
                    log("Best model is updated!")

                log("VALIDATION TARGET PREDICTIONS")

                metrics_pred_val_trg = algorithm.pred_meter_val_trg.get_metrics()
                """
                if args.task != "los":
                    log("ROC AUC score is : %.4f " % (metrics_pred_val_trg["roc_auc"]))
                    log("AUPRC score is : %.4f " % (metrics_pred_val_trg["avg_prc"]))
                else:
                    log("KAPPA score is : %.4f " % (metrics_pred_val_trg["kappa"]))
                """
                log_scores(args, dataset_type, metrics_pred_val_trg)

                #turn algorithm into training mode
                algorithm.train()

                #Refresh the saved metrics for algorithm
                algorithm.init_metrics()

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                end = time.time()
                
        else:
            continue
        break

# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='dann')

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99) #CLUDA
    parser.add_argument('-qs', '--queue_size', type=int, default=98304) #CLUDA
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true') #CLUDA
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--n_layers', type=int, default=1) #VRADA
    parser.add_argument('--h_dim', type=int, default=64) #VRADA
    parser.add_argument('--z_dim', type=int, default=64) #VRADA
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64') #All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3) #All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2) #All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1) #All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256) #All classifier and discriminators

    #The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=1.0)
    parser.add_argument('-w_kld', '--weight_KLD', type=float, default=1.0) #VRADA
    parser.add_argument('-w_nll', '--weight_NLL', type=float, default=1.0) #VRADA
    parser.add_argument('-w_cent', '--weight_cent_target', type=float, default=1.0) #CDAN
    #Below weights are defined for CLUDA
    parser.add_argument('--weight_loss_src', type=float, default=1.0)
    parser.add_argument('--weight_loss_trg', type=float, default=1.0)
    parser.add_argument('--weight_loss_ts', type=float, default=1.0)
    parser.add_argument('--weight_loss_disc', type=float, default=1.0)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')

    parser.add_argument('--path_src', type=str, default='../Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='../Data/aumc_fullstays')
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=int, default=1)
    parser.add_argument('--id_trg', type=int, default=2)

    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder))

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)