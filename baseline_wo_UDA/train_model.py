import sys
sys.path.append("..")
import os
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import ICUDataset, collate_test
from utils.augmentations import Augmenter
from utils.tcn_classifier import TCN_Classifier

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
    pred_loss = PredictionLoss(args.task, args.weight_ratio)

    log("Loss is weighted with " + str(pred_loss.loss_weights))

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_val_iteration = args.num_val_iteration

    #LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)

    dataset_src = ICUDataset(args.path_src, task=args.task, split_type="train", is_full_subset=True, is_cuda=True)
    dataset_val_src = ICUDataset(args.path_src, task=args.task, split_type="val", is_full_subset=True, is_cuda=True)
    dataset_test_src = ICUDataset(args.path_src, task=args.task, split_type="test", is_full_subset=True, is_cuda=True)


    dataset_trg = ICUDataset(args.path_trg, task=args.task, split_type="train", is_full_subset=True, is_cuda=True)
    dataset_val_trg = ICUDataset(args.path_trg, task=args.task, split_type="val", is_full_subset=True, is_cuda=True)
    dataset_test_trg = ICUDataset(args.path_trg, task=args.task, split_type="test", is_full_subset=True, is_cuda=True)

    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                            shuffle=True, num_workers=0)

    dataloader_trg= DataLoader(dataset_trg, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    dataloader_val_trg= DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                            shuffle=True, num_workers=0)

    #Check that we are not iterating more than the length of dataloaders
    assert num_val_iteration < len(dataloader_val_src)
    assert num_val_iteration < len(dataloader_val_trg)

    augmenter = Augmenter()

    #Calculate the output dim based on the task
    output_dim = 1 if args.task != "los" else 10

    model = TCN_Classifier(num_inputs=dataset_src[0]['sequence'].shape[1], output_dim=output_dim, num_channels=[64,64,64,64,64], num_static=dataset_src[0]['static'].shape[0], mlp_hidden_dim=256, use_batch_norm=args.use_batch_norm, use_mask=args.use_mask, kernel_size=3, dropout=args.dropout)

    model.cuda()


    #optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, amsgrad=False, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False, weight_decay=args.weight_decay, betas=(0.5, 0.99))
    current_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps)


    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    #Initialize progress metrics before training
    count_step = 0
    best_val_score= 0


    for i in range(args.num_epochs):
        total_loss = 0
        total_predictions = 0 
        for i_batch, sample_batched_src in enumerate(dataloader_src):
            #Go through augmentations first
            #seq, seq_mask = augmenter(sample_batched['sequence'], sample_batched['sequence_mask'])
            
            #model_output = model(seq, seq_mask, sample_batched['static'])
            
            model_output = model(sample_batched_src['sequence'], sample_batched_src['sequence_mask'], sample_batched_src['static'])

            loss = pred_loss.get_prediction_loss(model_output, sample_batched_src['label'])

            num_predictions = len(sample_batched_src['label'])
            total_predictions += num_predictions
            total_loss += loss.item()  * num_predictions

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_scheduler.step()
            
            count_step+=1
            
            #Log the progress and do evaluation
            if count_step%args.checkpoint_freq == 0:

                err = total_loss / total_predictions
                #Logging to tensorboard
                log("Train error at epoch %04d and step %06d : %.4f " % (i, count_step, err))

                total_loss = 0
                total_predictions = 0

                
                model.eval()

                pred_meter_val_src = PredictionMeter(args.task)

                total_val_loss = 0
                total_val_predictions = 0 

                y_val_list = []
                y_prob_out_list = []

                dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                        shuffle=True, num_workers=0)
                dataloader_val_src_iterator = iter(dataloader_val_src)

                for i_batch_val in range(args.num_val_iteration):

                    sample_batched_val_src = next(dataloader_val_src_iterator)

                    model_output = model(sample_batched_val_src['sequence'], sample_batched_val_src['sequence_mask'], sample_batched_val_src['static'])

                    loss = pred_loss.get_prediction_loss(model_output, sample_batched_src['label'])

                    num_predictions = len(sample_batched_val_src['label'])
                    total_val_predictions += num_predictions
                    total_val_loss += loss.item()  * num_predictions

                    pred_meter_val_src.update(sample_batched_val_src['label'], model_output)

                model.train()

                val_err = total_val_loss /total_val_predictions

                log("Validation error at epoch %04d and step %06d : %.4f " % (i, count_step, val_err))


                #Get the relevant prediction metrics
                metrics_pred_val_src = pred_meter_val_src.get_metrics()
                if args.task != "los":
                    log("Validation ROC AUC score at epoch %04d and step %06d : %.4f " % (i, count_step, metrics_pred_val_src["roc_auc"]))
                    log("Validation AUPRC score at epoch %04d and step %06d : %.4f " % (i, count_step, metrics_pred_val_src["avg_prc"]))
                else:
                    log("Validation KAPPA score at epoch %04d and step %06d : %.4f " % (i, count_step, metrics_pred_val_src["kappa"]))

                cur_val_score = metrics_pred_val_src["avg_prc"] if args.task != "los" else metrics_pred_val_src["kappa"]

                if cur_val_score > best_val_score:

                    model_best = pickle.loads(pickle.dumps(model))

                    torch.save({'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

                    best_val_score = cur_val_score
                    log("Best model is updated!")


    #DO EVALUATION
    log("SWITCHED TO EVALUATION")

    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=batch_size,
                        shuffle=False, num_workers=0)


    model_best.eval()

    pred_meter_test_trg = PredictionMeter(args.task)

    for i_batch, sample_batched in enumerate(dataloader_test_trg):
        #model_output = model_saved.predict(sample_batched['sequence'], sample_batched['static'], is_target=True)

        #seq, seq_mask = augmenter(sample_batched['sequence'], sample_batched['sequence_mask'])
        #model_output = model.predict(seq, sample_batched['static'], is_target=True)
        
        model_output = model_best(sample_batched['sequence'], sample_batched['sequence_mask'], sample_batched['static'])

        pred_meter_test_trg.update(sample_batched['label'], model_output, id_patient = sample_batched['patient_id'], stay_hour = sample_batched['stay_hour'])
        
        
    y_test_trg = np.array(pred_meter_test_trg.target_list)
    y_pred_trg = np.array(pred_meter_test_trg.output_list)
    id_test_trg = np.array(pred_meter_test_trg.id_patient_list)
    stay_hour_trg = np.array(pred_meter_test_trg.stay_hours_list)

    pred_trg_df = pd.DataFrame({"patient_id":id_test_trg, "stay_hour":stay_hour_trg, "y":y_test_trg, "y_pred":y_pred_trg})
    df_save_path_trg = os.path.join(args.experiments_main_folder, args.experiment_folder, "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)

    log("TARGET RESULTS")
    log("loaded from " + args.path_trg)
    log("")

    metrics_pred_test_trg = pred_meter_test_trg.get_metrics()
    if args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_trg["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_trg["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_trg["kappa"]))

    if args.task != "los":
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_trg = np.zeros(len(y_pred_trg))
            pred_label_trg[y_pred_trg>c] = 1

            acc_trg = accuracy_score(y_test_trg, pred_label_trg)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_trg))

    pred_meter_test_src = PredictionMeter(args.task)

    for i_batch, sample_batched in enumerate(dataloader_test_src):
        #model_output = model_saved.predict(sample_batched['sequence'], sample_batched['static'], is_target=True)

        #seq, seq_mask = augmenter(sample_batched['sequence'], sample_batched['sequence_mask'])
        #model_output = model.predict(seq, sample_batched['static'], is_target=True)
        
        model_output = model_best(sample_batched['sequence'], sample_batched['sequence_mask'], sample_batched['static'])

        pred_meter_test_src.update(sample_batched['label'], model_output, id_patient = sample_batched['patient_id'], stay_hour = sample_batched['stay_hour'])
        
        
    y_test_src = np.array(pred_meter_test_src.target_list)
    y_pred_src = np.array(pred_meter_test_src.output_list)
    id_test_src = np.array(pred_meter_test_src.id_patient_list)
    stay_hour_src = np.array(pred_meter_test_src.stay_hours_list)

    pred_src_df = pd.DataFrame({"patient_id":id_test_src, "stay_hour":stay_hour_src, "y":y_test_src, "y_pred":y_pred_src})
    df_save_path_src = os.path.join(args.experiments_main_folder, args.experiment_folder, "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)

    log("Source results saved to " + df_save_path_src)

    log("SOURCE RESULTS")
    log("loaded from " + args.path_src)
    log("")

    metrics_pred_test_src = pred_meter_test_src.get_metrics()
    if args.task != "los":
        log("Test ROC AUC score is : %.4f " % (metrics_pred_test_src["roc_auc"]))
        log("Test AUPRC score is : %.4f " % (metrics_pred_test_src["avg_prc"]))
    else:
        log("Test KAPPA score is : %.4f " % (metrics_pred_test_src["kappa"]))

    if args.task != "los":
        log("Accuracy scores for different thresholds: ")
        for c in np.arange(0.1,1,0.1):
            pred_label_src = np.zeros(len(y_pred_src))
            pred_label_src[y_pred_src>c] = 1

            acc_src = accuracy_score(y_test_src, pred_label_src)

            log("Test Accuracy for threshold %.2f : %.4f " % (c,acc_src))
            


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)


    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='DA_experiments')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--path_src', type=str, default='/local/home/oezyurty/TransferLearningICU/Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='/local/home/oezyurty/TransferLearningICU/Data/aumc_fullstays')
    parser.add_argument('--task', type=str, default='decompensation')

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder))

    with open(os.path.join(args.experiments_main_folder, args.experiment_folder, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)