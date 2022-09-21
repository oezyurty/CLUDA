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
    log = get_logger(os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "eval_"+saved_args.log))

    #Manually enter the loss weights for the task (for classification tasks)
    #It is to give more weight to minority class
    #loss_weights = (0.1, 0.9)

    #Some functions and variables for logging 
    dataset_type = get_dataset_type(saved_args)

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

    for i_batch, sample_batched in enumerate(dataloader_test_trg):
        algorithm.predict_trg(sample_batched)

    # even though the name is "pred_meter_val_trg", in this script it saves test results
    y_test_trg = np.array(algorithm.pred_meter_val_trg.target_list)
    y_pred_trg = np.array(algorithm.pred_meter_val_trg.output_list)
    id_test_trg = np.array(algorithm.pred_meter_val_trg.id_patient_list)
    stay_hour_trg = np.array(algorithm.pred_meter_val_trg.stay_hours_list)

    #If id_test and stay_hour are empty list, it means they are not kept for the current experiment. 
    #We will fill -1's as placholder
    if len(id_test_trg) == 0 and len(stay_hour_trg) == 0:
        id_test_trg = [-1] * len(y_test_trg)
        stay_hour_trg = [-1] * len(y_test_trg)

    pred_trg_df = pd.DataFrame({"patient_id":id_test_trg, "stay_hour":stay_hour_trg, "y":y_test_trg, "y_pred":y_pred_trg})
    df_save_path_trg = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)


    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")

    metrics_pred_test_trg = algorithm.pred_meter_val_trg.get_metrics()
    """
    if saved_args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_trg["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_trg["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_trg["kappa"]))
    """

    log_scores(saved_args, dataset_type, metrics_pred_test_trg)

    if algorithm.output_dim == 1:
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_trg = np.zeros(len(y_pred_trg))
            pred_label_trg[y_pred_trg>c] = 1

            acc_trg = accuracy_score(y_test_trg, pred_label_trg)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_trg))


    for i_batch, sample_batched in enumerate(dataloader_test_src):
        algorithm.predict_src(sample_batched)

    # even though the name is "pred_meter_val_src", in this script it saves test results
    y_test_src = np.array(algorithm.pred_meter_val_src.target_list)
    y_pred_src = np.array(algorithm.pred_meter_val_src.output_list)
    id_test_src = np.array(algorithm.pred_meter_val_src.id_patient_list)
    stay_hour_src = np.array(algorithm.pred_meter_val_src.stay_hours_list)

    #If id_test and stay_hour are empty list, it means they are not kept for the current experiment. 
    #We will fill -1's as placholder
    if len(id_test_src) == 0 and len(stay_hour_src) == 0:
        id_test_src = [-1] * len(y_test_src)
        stay_hour_src = [-1] * len(y_test_src)

    pred_src_df = pd.DataFrame({"patient_id":id_test_src, "stay_hour":stay_hour_src, "y":y_test_src, "y_pred":y_pred_src})
    df_save_path_src = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)

    log("Source results saved to " + df_save_path_src)

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    metrics_pred_test_src = algorithm.pred_meter_val_src.get_metrics()
    """
    if saved_args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_src["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_src["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_src["kappa"]))
    """

    log_scores(saved_args, dataset_type, metrics_pred_test_src)

    if algorithm.output_dim == 1:
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_src = np.zeros(len(y_pred_src))
            pred_label_src[y_pred_src>c] = 1

            acc_src = accuracy_score(y_test_src, pred_label_src)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_src))




# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')


    args = parser.parse_args()

    main(args)