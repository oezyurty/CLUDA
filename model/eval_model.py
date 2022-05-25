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

#from utils.mlp import MLP

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
    log = get_logger(os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "eval_"+saved_args.log))
    

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

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

    pred_meter_test_trg = PredictionMeter(saved_args.task)

    for i_batch, sample_batched in enumerate(dataloader_test_trg):

        seq = concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], saved_args.use_mask)
        
        model_output = model.predict(seq, sample_batched['static'], is_target=True)

        pred_meter_test_trg.update(sample_batched['label'], model_output, id_patient = sample_batched['patient_id'], stay_hour = sample_batched['stay_hour'])
        
    y_test_trg = np.array(pred_meter_test_trg.target_list)
    y_pred_trg = np.array(pred_meter_test_trg.output_list)
    id_test_trg = np.array(pred_meter_test_trg.id_patient_list)
    stay_hour_trg = np.array(pred_meter_test_trg.stay_hours_list)

    pred_trg_df = pd.DataFrame({"patient_id":id_test_trg, "stay_hour":stay_hour_trg, "y":y_test_trg, "y_pred":y_pred_trg})
    df_save_path_trg = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "predictions_test_target.csv")
    #df_save_path_trg = os.path.join(args.experiments_main_folder, args.experiment_folder, "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)

    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")

    metrics_pred_test_trg = pred_meter_test_trg.get_metrics()
    if saved_args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_trg["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_trg["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_trg["kappa"]))


    if saved_args.task != "los":
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_trg = np.zeros(len(y_pred_trg))
            pred_label_trg[y_pred_trg>c] = 1

            acc_trg = accuracy_score(y_test_trg, pred_label_trg)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_trg))



    pred_meter_test_src = PredictionMeter(saved_args.task)

    for i_batch, sample_batched in enumerate(dataloader_test_src):
        
        seq = concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], saved_args.use_mask)
        
        model_output = model.predict(seq, sample_batched['static'], is_target=False)

        pred_meter_test_src.update(sample_batched['label'], model_output, id_patient = sample_batched['patient_id'], stay_hour = sample_batched['stay_hour'])
        

    y_test_src = np.array(pred_meter_test_src.target_list)
    y_pred_src = np.array(pred_meter_test_src.output_list)
    id_test_src = np.array(pred_meter_test_src.id_patient_list)
    stay_hour_src = np.array(pred_meter_test_src.stay_hours_list)

    pred_src_df = pd.DataFrame({"patient_id":id_test_src, "stay_hour":stay_hour_src, "y":y_test_src, "y_pred":y_pred_src})
    df_save_path_src = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder, "predictions_test_source.csv")
    #df_save_path_src = os.path.join(args.experiments_main_folder, args.experiment_folder, "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)

    log("Source results saved to " + df_save_path_src)

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    metrics_pred_test_src = pred_meter_test_src.get_metrics()
    if saved_args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_src["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_src["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_src["kappa"]))

    if saved_args.task != "los":
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_src = np.zeros(len(y_pred_src))
            pred_label_src[y_pred_src>c] = 1

            acc_src = accuracy_score(y_test_src, pred_label_src)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_src))



# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='DA_experiments')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')


    args = parser.parse_args()

    main(args)
