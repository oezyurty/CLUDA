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

    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size

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


    num_val_iteration = len(dataloader_val_src)-1

    #Calculate the output dim based on the task
    output_dim = 1 if args.task != "los" else 10

    augmenter = Augmenter()

    model = CLUDA(num_inputs=(1+args.use_mask)*dataset_src[0]['sequence'].shape[1], output_dim=output_dim, num_channels=[64,64,64,64,64], num_static=dataset_src[0]['static'].shape[0], 
        use_static=False, mlp_hidden_dim=256, use_batch_norm=args.use_batch_norm, kernel_size=3, dropout=args.dropout, K=args.queue_size, m=args.momentum)

    model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.99), amsgrad=False, weight_decay=args.weight_decay)
    current_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=args.warmup_steps)


    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder)

    #Initialize SummaryWriter
    writer = SummaryWriter(experiment_folder_path)

    #Initialize progress metrics before training
    count_step = 0
    best_val_score = 0

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_s = AverageMeter('Loss Source', ':.4e')
    top1_s = AverageMeter('Acc@1', ':6.2f')
    losses_t = AverageMeter('Loss Target', ':.4e')
    top1_t = AverageMeter('Acc@1', ':6.2f')
    losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
    top1_ts = AverageMeter('Acc@1', ':6.2f')
    losses_disc = AverageMeter('Loss Sour-Tar Disc', ':.4e')
    top1_disc = AverageMeter('Acc@1', ':6.2f')
    losses_pred = AverageMeter('Loss Pred', ':.4e')
    score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
    losses = AverageMeter('Loss TOTAL', ':.4e')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(dataloader_src),
        [batch_time, data_time, losses_s, top1_s, losses_t, top1_t, losses_ts, top1_ts, losses_disc, top1_disc, losses_pred, score_pred, losses],
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

            #p is a linear term (between 0 and 1) for the progress of training.
            p = float(count_step) / (2 * len(dataloader_src))
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            #Go through augmentations first
            seq_q_src, seq_mask_q_src = augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])
            seq_k_src, seq_mask_k_src = augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])
            
            seq_q_trg, seq_mask_q_trg = augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])
            seq_k_trg, seq_mask_k_trg = augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])

            #Concat mask if use_mask = True
            seq_q_src = concat_mask(seq_q_src, seq_mask_q_src, args.use_mask)
            seq_k_src = concat_mask(seq_k_src, seq_mask_k_src, args.use_mask)
            seq_q_trg = concat_mask(seq_q_trg, seq_mask_q_trg, args.use_mask)
            seq_k_trg = concat_mask(seq_k_trg, seq_mask_k_trg, args.use_mask)
            
            # compute output
            output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s = model(seq_q_src, seq_k_src, sample_batched_src['static'], seq_q_trg, seq_k_trg, sample_batched_trg['static'], alpha)
            
            # Compute all losses
            loss_s = criterion(output_s, target_s)
            loss_t = criterion(output_t, target_t)
            loss_ts = criterion(output_ts, target_ts)
            loss_disc = F.binary_cross_entropy(output_disc, target_disc)
            
            
            # Task classification  Loss
            loss_pred = pred_loss.get_prediction_loss(pred_s, sample_batched_src['label'])
            
            # Calculate overall loss
            loss = args.weight_loss_src*loss_s + args.weight_loss_trg*loss_t + \
                    args.weight_loss_ts*loss_ts + args.weight_loss_disc*loss_disc + args.weight_loss_pred*loss_pred
            
            
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            acc1 = accuracy(output_s, target_s, topk=(1, ))
            losses_s.update(loss_s.item(), seq_q_src.size(0))
            top1_s.update(acc1[0][0], seq_q_src.size(0))
            
            acc1 = accuracy(output_t, target_t, topk=(1, ))
            losses_t.update(loss_t.item(), seq_q_trg.size(0))
            top1_t.update(acc1[0][0], seq_q_trg.size(0))

            acc1 = accuracy(output_ts, target_ts, topk=(1, ))
            losses_ts.update(loss_t.item(), seq_q_trg.size(0))
            top1_ts.update(acc1[0][0], seq_q_trg.size(0))
            
            acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten()>0.5, target_disc.detach().cpu().numpy().flatten())
            losses_disc.update(loss_disc.item(), output_disc.size(0))
            top1_disc.update(acc1, output_disc.size(0))
            
            acc1 = accuracy(pred_s, sample_batched_src['label'], topk=(1, ))
            losses_pred.update(loss_pred.item(), seq_q_src.size(0))

            pred_meter_src = PredictionMeter(args.task)

            pred_meter_src.update(sample_batched_src['label'], pred_s)

            metrics_pred_src = pred_meter_src.get_metrics()

            if args.task != "los":
                score_pred.update(metrics_pred_src["roc_auc"],sample_batched_src['sequence'].size(0))
            else:
                score_pred.update(metrics_pred_src["kappa"],sample_batched_src['sequence'].size(0))
            
            
            losses.update(loss.item(), seq_q_src.size(0))
        

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_scheduler.step()
            
            count_step+=1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            
            
            if count_step % args.checkpoint_freq == 0:
            
                log(progress.display(i_batch+1, is_logged=True))
                
                #Write to tensorboard
                write_to_tensorboard(writer, progress, count_step, split_type="train", task=args.task)

                model.eval()

                pred_meter_val_src = PredictionMeter(args.task)
                pred_meter_val_trg = PredictionMeter(args.task)

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_s = AverageMeter('Loss Source', ':.4e')
                top1_s = AverageMeter('Acc@1', ':6.2f')
                losses_t = AverageMeter('Loss Target', ':.4e')
                top1_t = AverageMeter('Acc@1', ':6.2f')
                losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
                top1_ts = AverageMeter('Acc@1', ':6.2f')
                losses_disc = AverageMeter('Loss Sour-Tar Disc', ':.4e')
                top1_disc = AverageMeter('Acc@1', ':6.2f')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                #top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    num_val_iteration,
                    [batch_time, data_time, losses_s, top1_s, losses_t, top1_t, losses_ts, top1_ts, losses_disc, top1_disc, losses_pred, score_pred, losses],
                    prefix="Epoch: [{}]".format(i))

                end = time.time()
                
                #Keep prediction results of the Source
                y_val_src_list = []
                y_prob_out_src_list = []
                #Keep prediction results of the Target
                y_val_trg_list = []
                y_prob_out_trg_list = []
                
                
                flag_eval_trg = True
                
                dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0)
                dataloader_val_trg_iterator = iter(dataloader_val_trg)

                for i_batch_val, sample_batched_val_src in enumerate(dataloader_val_src):
                    #Current model does not support smaller batches than batch_size (due to queue ptr)
                    if len(sample_batched_val_src['sequence']) != eval_batch_size:
                        continue
                        
                    #Since AUMC is a smaller dataset, if we are out of batch, initialize iterator once again.
                    try:
                        sample_batched_val_trg = next(dataloader_val_trg_iterator)
                    except StopIteration:
                        dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                            shuffle=True, num_workers=0)
                        dataloader_val_trg_iterator = iter(dataloader_val_trg)
                        sample_batched_val_trg = next(dataloader_val_trg_iterator)

                        flag_eval_trg = False
                    
                    #Current model does not support smaller batches than batch_size (due to queue ptr)
                    if len(sample_batched_val_trg['sequence']) != eval_batch_size:
                        flag_eval_trg = False
                        continue


                    # measure data loading time
                    data_time.update(time.time() - end)
                    
                    #Go through augmentations first
                    seq_q_src, seq_mask_q_src = augmenter(sample_batched_val_src['sequence'], sample_batched_val_src['sequence_mask'])
                    seq_k_src, seq_mask_k_src = augmenter(sample_batched_val_src['sequence'], sample_batched_val_src['sequence_mask'])

                    seq_q_trg, seq_mask_q_trg = augmenter(sample_batched_val_trg['sequence'], sample_batched_val_trg['sequence_mask'])
                    seq_k_trg, seq_mask_k_trg = augmenter(sample_batched_val_trg['sequence'], sample_batched_val_trg['sequence_mask'])

                    #Concat mask if use_mask = True
                    seq_q_src = concat_mask(seq_q_src, seq_mask_q_src, args.use_mask)
                    seq_k_src = concat_mask(seq_k_src, seq_mask_k_src, args.use_mask)
                    seq_q_trg = concat_mask(seq_q_trg, seq_mask_q_trg, args.use_mask)
                    seq_k_trg = concat_mask(seq_k_trg, seq_mask_k_trg, args.use_mask)

                    # compute output
                    output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s = model(seq_q_src, seq_k_src, sample_batched_val_src['static'], seq_q_trg, seq_k_trg, sample_batched_val_trg['static'], count_step)

                    # Compute all losses
                    loss_s = criterion(output_s, target_s)
                    loss_t = criterion(output_t, target_t)
                    loss_ts = criterion(output_ts, target_ts)
                    loss_disc = F.binary_cross_entropy(output_disc, target_disc)

                    
                    # Task classification  Loss
                    loss_pred = pred_loss.get_prediction_loss(pred_s, sample_batched_val_src['label'])

                    #Calculate overall loss
                    loss = args.weight_loss_src*loss_s + args.weight_loss_trg*loss_t + \
                            args.weight_loss_ts*loss_ts + args.weight_loss_disc*loss_disc + args.weight_loss_pred*loss_pred
                    

                    # acc1/acc5 are (K+1)-way contrast classifier accuracy
                    # measure accuracy and record loss
                    acc1 = accuracy(output_s, target_s, topk=(1, ))
                    losses_s.update(loss_s.item(), seq_q_src.size(0))
                    top1_s.update(acc1[0][0], seq_q_src.size(0))

                    acc1 = accuracy(output_t, target_t, topk=(1, ))
                    losses_t.update(loss_t.item(), seq_q_trg.size(0))
                    top1_t.update(acc1[0][0], seq_q_trg.size(0))

                    acc1 = accuracy(output_ts, target_ts, topk=(1, ))
                    losses_ts.update(loss_t.item(), seq_q_trg.size(0))
                    top1_ts.update(acc1[0][0], seq_q_trg.size(0))
                    
                    acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten()>0.5, target_disc.detach().cpu().numpy().flatten())
                    losses_disc.update(loss_disc.item(), output_disc.size(0))
                    top1_disc.update(acc1, output_disc.size(0))

                    acc1 = accuracy(pred_s, sample_batched_val_src['label'], topk=(1, ))
                    losses_pred.update(loss_pred.item(), seq_q_src.size(0))

                    pred_meter_val= PredictionMeter(args.task)

                    pred_meter_val.update(sample_batched_val_src['label'], pred_s)

                    metrics_pred_val= pred_meter_val.get_metrics()

                    if args.task != "los":
                        score_pred.update(metrics_pred_val["roc_auc"],sample_batched_val_src['sequence'].size(0))
                    else:
                        score_pred.update(metrics_pred_val["kappa"],sample_batched_val_src['sequence'].size(0))

    

                    losses.update(loss.item(), seq_q_src.size(0))
                    
                    #keep track of prediction results (of source) explicitly

                    pred_meter_val_src.update(sample_batched_val_src['label'], pred_s)
                    
                    #keep track of prediction results (of target) explicitly
                    if flag_eval_trg:
                        y_pred_trg = model.predict(seq_q_trg, sample_batched_val_trg['static'], is_target=True)
                        

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
                    model_best = pickle.loads(pickle.dumps(model))

                    torch.save({'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

                    best_val_score = cur_val_score
                    log("Best model is updated!")
                    
                log("VALIDATION TARGET PREDICTIONS")

                metrics_pred_val_trg = pred_meter_val_trg.get_metrics()
                if args.task != "los":
                    log("ROC AUC score is : %.4f " % (metrics_pred_val_trg["roc_auc"]))
                    log("AUPRC score is : %.4f " % (metrics_pred_val_trg["avg_prc"]))
                    write_to_tensorboard(writer, progress, count_step, split_type="val", task=args.task, 
                        auroc_s=metrics_pred_val_src["roc_auc"], auprc_s=metrics_pred_val_src["avg_prc"], 
                        auroc_t=metrics_pred_val_trg["roc_auc"], auprc_t=metrics_pred_val_trg["avg_prc"])
                else:
                    log("KAPPA score is : %.4f " % (metrics_pred_val_trg["kappa"]))
                    write_to_tensorboard(writer, progress, count_step, split_type="val", task=args.task, 
                        kappa_s=metrics_pred_val_src["kappa"], kappa_t=metrics_pred_val_trg["kappa"])
                
                model.train()
                    
                
                
                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses_s = AverageMeter('Loss Source', ':.4e')
                top1_s = AverageMeter('Acc@1', ':6.2f')
                losses_t = AverageMeter('Loss Target', ':.4e')
                top1_t = AverageMeter('Acc@1', ':6.2f')
                losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
                top1_ts = AverageMeter('Acc@1', ':6.2f')
                losses_disc = AverageMeter('Loss Sour-Tar Disc', ':.4e')
                top1_disc = AverageMeter('Acc@1', ':6.2f')
                losses_pred = AverageMeter('Loss Pred', ':.4e')
                score_pred = AverageMeter('ROC AUC', ':6.2f') if args.task != "los" else AverageMeter('Kappa', ':6.2f')
                losses = AverageMeter('Loss TOTAL', ':.4e')
                #top5 = AverageMeter('Acc@5', ':6.2f')
                progress = ProgressMeter(
                    len(dataloader_src),
                    [batch_time, data_time, losses_s, top1_s, losses_t, top1_t, losses_ts, top1_ts, losses_disc, top1_disc, losses_pred, score_pred, losses],
                    prefix="Epoch: [{}]".format(i))

                end = time.time()


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)
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

    #Training weights for different loss components
    parser.add_argument('--weight_loss_src', type=float, default=1.0)
    parser.add_argument('--weight_loss_trg', type=float, default=1.0)
    parser.add_argument('--weight_loss_ts', type=float, default=1.0)
    parser.add_argument('--weight_loss_disc', type=float, default=1.0)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='DA_experiments')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('-l', '--log', type=str, default='train.log')

    parser.add_argument('--path_src', type=str, default='../Data/miiv_fullstays')
    parser.add_argument('--path_trg', type=str, default='../Data/aumc_fullstays')
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
