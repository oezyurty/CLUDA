import sys
sys.path.append("..")

import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import ICUDataset
from utils.augmentations import Augmenter
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet
from utils.util_progress_log import ProgressMeter, AverageMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter
from utils.loss import PredictionLoss

from models.models import ReverseLayerF, SASA
from models.loss import MMD_loss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import time
import shutil

import pickle
import json
import logging

from argparse import ArgumentParser

import pdb

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

    #Manually enter the loss weights for the task (for classification tasks)
    #It is to give more weight to minority class
    #loss_weights = (0.1, 0.9)

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

    #Calculate the output dim based on the task
    output_dim = 1 if args.task != "los" else 10

    feature_extractor = SASA(x_dim=dataset_src[0]['sequence'].shape[1], h_dim=args.h_dim, n_segments=args.n_segments, seg_length=args.seg_length)

    classifier = MLP(input_dim = args.h_dim * 2 * dataset_src[0]['sequence'].shape[1]  + dataset_src[0]['static'].shape[0], hidden_dim = 256, 
                                   output_dim = output_dim, use_batch_norm = args.use_batch_norm)

    network = nn.Sequential(feature_extractor, classifier)


    feature_extractor.cuda()
    classifier.cuda()

    mmd_loss_calc = MMD_loss(kernel_type='linear', kernel_num=1)

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
    )

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    #Initialize progress metrics before training
    count_step = 0
    best_val_score = 0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
    losses_pred = AverageMeter('Loss Pred', ':.4e')
    score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
    losses = AverageMeter('Loss TOTAL', ':.4e')
    progress = ProgressMeter(
        len(dataloader_src),
        [batch_time, data_time, losses_ts, losses_pred, score_pred, losses],
        prefix="Epoch: [{}]".format(0))

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

            src_att_intra, src_att_inter, src_feat = feature_extractor(sample_batched_src['sequence'])
            trg_att_intra, trg_att_inter, trg_feat = feature_extractor(sample_batched_trg['sequence'])

            src_pred = classifier(src_feat, sample_batched_src['static'])

            #Calculate intra mmd loss
            list_loss_intra = [mmd_loss_calc(src_att_intra[i], trg_att_intra[i]) for i in range(len(src_att_intra))]
            #loss_intra = sum(list_loss_intra) / len(list_loss_intra)
            loss_intra = sum(list_loss_intra)
            #loss_intra.requires_grad = True

            #Calculate inter mmd loss
            list_loss_inter = [mmd_loss_calc(src_att_inter[i], trg_att_inter[i]) for i in range(len(src_att_inter))]
            #loss_inter = sum(list_loss_inter) / len(list_loss_inter)
            loss_inter = sum(list_loss_inter)
            #loss_inter.requires_grad = True

            # Task classification  Loss
            src_cls_loss = pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

            loss =  args.weight_intra*loss_intra + args.weight_inter*loss_inter + src_cls_loss

            # zero grad
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
               
            
            # measure domain loss
            losses_ts.update(loss_intra.item() + loss_inter.item(), sample_batched_src['sequence'].size(0))
            
            losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

            pred_meter_src = PredictionMeter(args.task)

            pred_meter_src.update(sample_batched_src['label'], src_pred)

            metrics_pred_src = pred_meter_src.get_metrics()

            if args.task != "los":
                score_pred.update(metrics_pred_src["roc_auc"],sample_batched_src['sequence'].size(0))
            else:
                score_pred.update(metrics_pred_src["kappa"],sample_batched_src['sequence'].size(0))

            losses.update(loss.item(), sample_batched_src['sequence'].size(0))
        
            
            count_step+=1


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if count_step % args.checkpoint_freq == 0:
            
                log(progress.display(i_batch+1, is_logged=True))
                
                #Write to tensorboard
                #write_to_tensorboard(writer, progress, count_step, split_type="train")

                feature_extractor.eval()
                classifier.eval()

                pred_meter_val_src = PredictionMeter(args.task)
                pred_meter_val_trg = PredictionMeter(args.task)

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                progress = ProgressMeter(
                    args.num_val_iteration,
                    [batch_time, data_time, losses_ts, losses_pred, score_pred, losses],
                    prefix="Epoch: [{}]".format(i))

                end = time.time()
                
                #Keep prediction results of the Source
                y_val_src_list = []
                y_prob_out_src_list = []
                #Keep prediction results of the Target
                y_val_trg_list = []
                y_prob_out_trg_list = []
                
                dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_val_src_iterator = iter(dataloader_val_src)
                
                dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_val_trg_iterator = iter(dataloader_val_trg)
                
                
                for i_batch_val in range(args.num_val_iteration):
                    sample_batched_val_src = next(dataloader_val_src_iterator)
                    sample_batched_val_trg = next(dataloader_val_trg_iterator)


                    # measure data loading time
                    data_time.update(time.time() - end)


                    src_att_intra, src_att_inter, src_feat = feature_extractor(sample_batched_val_src['sequence'])
                    trg_att_intra, trg_att_inter, trg_feat = feature_extractor(sample_batched_val_trg['sequence'])

                    src_pred = classifier(src_feat, sample_batched_val_src['static'])

                    #Calculate intra mmd loss
                    list_loss_intra = [mmd_loss_calc(src_att_intra[i], trg_att_intra[i]) for i in range(len(src_att_intra))]
                    #loss_intra = sum(list_loss_intra) / len(list_loss_intra)
                    loss_intra = sum(list_loss_intra)

                    #Calculate inter mmd loss
                    list_loss_inter = [mmd_loss_calc(src_att_inter[i], trg_att_inter[i]) for i in range(len(src_att_inter))]
                    #loss_inter = sum(list_loss_inter) / len(list_loss_inter)
                    loss_inter = sum(list_loss_inter)

                    # Task classification  Loss
                    src_cls_loss = pred_loss.get_prediction_loss(src_pred, sample_batched_val_src['label'])

                    loss =  args.weight_intra*loss_intra + args.weight_inter*loss_inter + src_cls_loss


                    # measure domainloss
                    losses_ts.update(loss_intra.item() + loss_inter.item(), sample_batched_val_src['sequence'].size(0))

                    losses_pred.update(src_cls_loss.item(), sample_batched_val_src['sequence'].size(0))

                    pred_meter_val= PredictionMeter(args.task)

                    pred_meter_val.update(sample_batched_val_src['label'], src_pred)

                    metrics_pred_val= pred_meter_val.get_metrics()

                    if args.task != "los":
                        score_pred.update(metrics_pred_val["roc_auc"],sample_batched_val_src['sequence'].size(0))
                    else:
                        score_pred.update(metrics_pred_val["kappa"],sample_batched_val_src['sequence'].size(0))

                    losses.update(loss.item(), sample_batched_val_src['sequence'].size(0))
                    
                    #keep track of prediction results (of source) explicitly
                    pred_meter_val_src.update(sample_batched_val_src['label'], src_pred)


                    #keep track of prediction results (of target) explicitly
                    y_pred_trg = classifier(trg_feat, sample_batched_val_trg['static'])

                    pred_meter_val_trg.update(sample_batched_val_trg['label'], y_pred_trg)
                    
                log("VALIDATION RESULTS")
                log(progress.display(i_batch_val+1, is_logged=True))
                
                log("VALIDATION SOURCE PREDICTIONS")

                metrics_pred_val_src = pred_meter_val_src.get_metrics()
                if args.task != "los":
                    log("ROC AUC score is : %.4f " % (metrics_pred_val_src["roc_auc"]))
                    log("AUPRC score is : %.4f " % (metrics_pred_val_src["avg_prc"]))
                else:
                    log("KAPPA score is : %.4f " % (metrics_pred_val_src["kappa"]))

                cur_val_score = metrics_pred_val_src["avg_prc"] if args.task != "los" else metrics_pred_val_src["kappa"]
                
                if cur_val_score > best_val_score:

                    torch.save({'feature_extractor_state_dict': feature_extractor.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'optimizer' : optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

                    best_val_score = cur_val_score
                    log("Best model is updated!")
                    
                log("VALIDATION TARGET PREDICTIONS")

                metrics_pred_val_trg = pred_meter_val_trg.get_metrics()
                if args.task != "los":
                    log("ROC AUC score is : %.4f " % (metrics_pred_val_trg["roc_auc"]))
                    log("AUPRC score is : %.4f " % (metrics_pred_val_trg["avg_prc"]))
                else:
                    log("KAPPA score is : %.4f " % (metrics_pred_val_trg["kappa"]))
                
                feature_extractor.train()
                classifier.train()
                    
                
                
                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                progress = ProgressMeter(
                    len(dataloader_src),
                    [batch_time, data_time, losses_ts, losses_pred, score_pred, losses],
                    prefix="Epoch: [{}]".format(i))

                end = time.time()


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=20)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)

    #SASA specific param
    parser.add_argument('--n_segments', type=int, default=5)
    parser.add_argument('--seg_length', type=int, default=4)
    parser.add_argument('--h_dim', type=int, default=2)

    #The weight of the domain classification loss
    parser.add_argument('-w_intra', '--weight_intra', type=float, default=1.0)
    parser.add_argument('-w_inter', '--weight_inter', type=float, default=1.0)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='experiments_DANN')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')

    parser.add_argument('--path_src', type=str, default='/local/home/oezyurty/TransferLearningICU/Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='/local/home/oezyurty/TransferLearningICU/Data/aumc_fullstays')
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
