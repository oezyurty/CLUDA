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
from utils.tcn import TemporalConvNet
from utils.util_progress_log import ProgressMeter, AverageMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter
from utils.loss import PredictionLoss

from models.models import ReverseLayerF
from models.loss import ConditionalEntropyLoss

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

    #Calculate the output dim based on the task
    output_dim = 1 if args.task != "los" else 10

    feature_extractor = TemporalConvNet(num_inputs=dataset_src[0]['sequence'].shape[1], num_channels=[64,64,64,64,64], kernel_size=3, dropout=args.dropout)

    classifier = MLP(input_dim = 64 + dataset_src[0]['static'].shape[0], hidden_dim = 256, 
                                   output_dim = output_dim, use_batch_norm = args.use_batch_norm)

    domain_classifier = MLP(input_dim = 64 * output_dim, hidden_dim = 256, 
                                   output_dim = 1, use_batch_norm = args.use_batch_norm)

    network = nn.Sequential(feature_extractor, classifier)

    criterion_cond = ConditionalEntropyLoss(args.task)


    feature_extractor.cuda()
    classifier.cuda()
    domain_classifier.cuda()
    criterion_cond.cuda()

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
    )
    optimizer_disc = torch.optim.Adam(
        domain_classifier.parameters(),
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
    top1_ts = AverageMeter('Acc@1', ':6.2f')
    lossess_cent = AverageMeter('Loss CENT', ':.4e')
    losses_pred = AverageMeter('Loss Pred', ':.4e')
    score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
    losses = AverageMeter('Loss TOTAL', ':.4e')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader_src),
        [batch_time, data_time, losses_ts, top1_ts, lossess_cent, losses_pred, score_pred, losses],
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

            p = float(count_step) / 400
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            domain_label_src = torch.ones(len(sample_batched_src['sequence'])).cuda()
            domain_label_trg = torch.zeros(len(sample_batched_trg['sequence'])).cuda()
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
            
            src_feat = feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
            src_pred = classifier(src_feat, sample_batched_src['static'])

            trg_feat = feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]
            trg_pred = classifier(trg_feat, sample_batched_trg['static'])

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
            disc_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)

            # update Domain classification
            optimizer_disc.zero_grad()
            disc_loss.backward()
            optimizer_disc.step()


            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(sample_batched_src['sequence'])).cuda()
            domain_label_trg = torch.ones(len(sample_batched_trg['sequence'])).cuda()
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

             # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
            # loss of domain discriminator according to fake labels
            domain_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)

            # Task classification  Loss
            src_cls_loss = pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

            # conditional entropy loss.
            loss_trg_cent = criterion_cond(trg_pred)

            # total loss
            loss = src_cls_loss + args.weight_domain*domain_loss + args.weight_cent_target*loss_trg_cent

            # update feature extractor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
               
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1 = accuracy_score(disc_prediction.detach().cpu().numpy().flatten()>0.5, 
                            domain_label_concat.detach().cpu().numpy().flatten())
            losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
            top1_ts.update(acc1, 2*sample_batched_src['sequence'].size(0))

            lossess_cent.update(loss_trg_cent.item(), sample_batched_trg['sequence'].size(0))
            
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
                domain_classifier.eval()
                classifier.eval()

                pred_meter_val_src = PredictionMeter(args.task)
                pred_meter_val_trg = PredictionMeter(args.task)

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
                top1_ts = AverageMeter('Acc@1', ':6.2f')
                lossess_cent = AverageMeter('Loss CENT', ':.4e')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                #top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    args.num_val_iteration,
                    [batch_time, data_time, losses_ts, top1_ts, lossess_cent, losses_pred, score_pred, losses],
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

                    p = float(count_step) / 400
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1


                    domain_label_src = torch.ones(len(sample_batched_val_src['sequence'])).cuda()
                    domain_label_trg = torch.zeros(len(sample_batched_val_trg['sequence'])).cuda()
                    domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)
                    
                    src_feat = feature_extractor(sample_batched_val_src['sequence'].transpose(1,2))[:,:,-1]
                    src_pred = classifier(src_feat, sample_batched_val_src['static'])

                    trg_feat = feature_extractor(sample_batched_val_trg['sequence'].transpose(1,2))[:,:,-1]
                    trg_pred = classifier(trg_feat, sample_batched_val_trg['static'])

                    # concatenate features and predictions
                    feat_concat = torch.cat((src_feat, trg_feat), dim=0)
                    pred_concat = torch.cat((src_pred, trg_pred), dim=0)

                    # Domain classification loss
                    feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
                    disc_prediction = domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
                    disc_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)



                    # prepare fake domain labels for training the feature extractor
                    domain_label_src = torch.zeros(len(sample_batched_val_src['sequence'])).cuda()
                    domain_label_trg = torch.ones(len(sample_batched_val_trg['sequence'])).cuda()
                    domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

                     # Repeat predictions after updating discriminator
                    feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
                    disc_prediction = domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
                    # loss of domain discriminator according to fake labels
                    domain_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)

                    # Task classification  Loss
                    src_cls_loss = pred_loss.get_prediction_loss(src_pred, sample_batched_val_src['label'])

                    # conditional entropy loss.
                    loss_trg_cent = criterion_cond(trg_pred)

                    # total loss
                    loss = src_cls_loss + args.weight_domain*domain_loss + args.weight_cent_target*loss_trg_cent


                    # acc1/acc5 are (K+1)-way contrast classifier accuracy
                    # measure accuracy and record loss
                    acc1 = accuracy_score(disc_prediction.detach().cpu().numpy().flatten()>0.5, 
                                    domain_label_concat.detach().cpu().numpy().flatten())
                    losses_ts.update(domain_loss.item(), 2*sample_batched_val_src['sequence'].size(0))
                    top1_ts.update(acc1, 2*sample_batched_val_src['sequence'].size(0))

                    lossess_cent.update(loss_trg_cent.item(), sample_batched_trg['sequence'].size(0))
                    
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
                    pred_meter_val_trg.update(sample_batched_val_trg['label'], trg_pred)

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
                        'domain_classifier_state_dict': domain_classifier.state_dict(),
                        'classifier_state_dict': classifier.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'optimizer_disc' : optimizer_disc.state_dict()}, 
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
                domain_classifier.train()
                classifier.train()
                    
                
                
                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
                top1_ts = AverageMeter('Acc@1', ':6.2f')
                lossess_cent = AverageMeter('Loss CENT', ':.4e')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                #top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    len(dataloader_src),
                    [batch_time, data_time, losses_ts, top1_ts, lossess_cent, losses_pred, score_pred, losses],
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
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-2)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)

    #The weight of the domain classification loss (and CEnt on target)
    parser.add_argument('-w_d', '--weight_domain', type=float, default=1.0)
    parser.add_argument('-w_cent', '--weight_cent_target', type=float, default=1.0)


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
