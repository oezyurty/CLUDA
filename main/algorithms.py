import sys
sys.path.append("..")
import os
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils.dataset import get_output_dim
from utils.augmentations import Augmenter
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet
from utils.augmentations import Augmenter, concat_mask

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util_progress_log import AverageMeter, ProgressMeter, accuracy, write_to_tensorboard, get_logger, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss

from models.models import ReverseLayerF, VRNN, AdvSKM_Disc
from models.loss import MMD_loss, ConditionalEntropyLoss, CORAL, LMMD_loss, HoMM_loss
from models.cluda import DA_MoCoNNQQ_Disc_TCN_Siam


import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import time
import shutil

import pickle
import json
import logging

from argparse import ArgumentParser
from collections import namedtuple


#Given the args, it will return the algorithm directly
def get_algorithm(args, input_channels_dim, input_static_dim):
    if args.algo_name == "codats":
        return CoDATS(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "vrada":
        return VRADA(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "advskm":
        return AdvSKM(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "cdan":
        return CDAN(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "ddc":
        return DDC(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "can":
        return CAN(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "deepcoral":
        return DeepCORAL(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "dsan":
        return DSAN(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "homm":
        return HoMM(args, input_channels_dim, input_static_dim)
    elif args.algo_name == "mmda":
        return MMDA(args, input_channels_dim, input_static_dim)
    #w/o UDA algorithm below
    elif args.algo_name == "tcn":
        return TCN(args, input_channels_dim, input_static_dim)
    #CLUDA algorithm below
    elif args.algo_name == "cluda":
        return CLUDA(args, input_channels_dim, input_static_dim)
    else:
        return None


#Helper function for algorithms to get a list of channels (i.e. For TCN) from args
def get_num_channels(args):
    return list(map(int, args.num_channels_TCN.split("-")))

class Base_Algorithm(nn.Module):
    def __init__(self, args):
        super(Base_Algorithm, self).__init__()

        #Record the args if needed later on
        self.args = args

        #Let algorithm know its name and dataset_type
        self.algo_name = args.algo_name
        self.dataset_type = get_dataset_type(args)

        self.pred_loss = PredictionLoss(self.dataset_type, args.task, args.weight_ratio)

        self.output_dim = get_output_dim(args)

        #Only used for TCN-related models
        self.num_channels = get_num_channels(args)

        #During training we report one main metric
        self.main_pred_metric = ""
        if self.dataset_type == "icu":
            if args.task != "los":
                self.main_pred_metric = "roc_auc"
            else:
                self.main_pred_metric = "kappa"
        #If it is sensor data, we will use Macro f1
        else:
            self.main_pred_metric = "mac_f1"


        self.init_pred_meters_val()

    def init_pred_meters_val(self):
            #We save prediction scores for validation set (for reporting purposes)
            self.pred_meter_val_src = PredictionMeter(self.args)
            self.pred_meter_val_trg = PredictionMeter(self.args)

    def predict_trg(self, sample_batched):
        trg_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_trg = self.classifier(trg_feat, sample_batched.get('static'))

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):
        src_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_src = self.classifier(src_feat, sample_batched.get('static'))

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))
    
    def get_embedding(self, sample_batched):
        feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        return feat

    #Score prediction is dataset and task dependent, that's why we write init function here
    def init_score_pred(self):
        if self.dataset_type == "icu":
            if self.args.task != "los":
                return AverageMeter('ROC AUC', ':6.2f')
            else:
                return AverageMeter('Kappa', ':6.2f')
        #If it is sensor data, we will use Macro f1
        else:
            return AverageMeter('Macro F1', ':6.2f')

    #Helper function to build TCN feature extractor for all related algorithms 
    def build_feature_extractor_TCN(self, args, input_channels_dim, num_channels):
        return TemporalConvNet(num_inputs=input_channels_dim, num_channels=num_channels, kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout)

#CoDATS Algorithm Implementation
class CoDATS(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(CoDATS, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)
        self.domain_classifier = MLP(input_dim = self.feature_extractor.num_channels[-1], hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = 1, use_batch_norm = args.use_batch_norm)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
        self.domain_classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        p = float(kwargs.get("count_step")) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        domain_label_src = torch.ones(len(sample_batched_src['sequence'])).cuda()
        domain_label_trg = torch.zeros(len(sample_batched_trg['sequence'])).cuda()

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed, None)
        src_domain_loss = F.binary_cross_entropy(src_domain_pred.flatten(), domain_label_src)

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed, None)
        trg_domain_loss =F.binary_cross_entropy(trg_domain_pred.flatten(), domain_label_trg)

        # Total domain loss
        domain_loss = (src_domain_loss + trg_domain_loss)/2

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy_score(torch.cat([src_domain_pred, trg_domain_pred]).detach().cpu().numpy().flatten()>0.5, 
                        torch.cat([domain_label_src, domain_label_trg]).detach().cpu().numpy().flatten())
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        self.top1_ts.update(acc1, 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.top1_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'domain_classifier_state_dict': self.domain_classifier.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'optimizer_disc' : self.optimizer_disc.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])


#VRADA Algorithm Implementation
class VRADA(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(VRADA, self).__init__(args)

        self.feature_extractor = VRNN(x_dim=input_channels_dim, h_dim=args.h_dim, z_dim=args.z_dim, n_layers=args.n_layers)
        self.classifier = MLP(input_dim = args.z_dim + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)
        self.domain_classifier = MLP(input_dim = args.z_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = 1, use_batch_norm = args.use_batch_norm)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
        self.domain_classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()

    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        p = float(kwargs.get("count_step")) / 400
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        domain_label_src = torch.ones(len(sample_batched_src['sequence'])).cuda()
        domain_label_trg = torch.zeros(len(sample_batched_trg['sequence'])).cuda()

        src_kld_loss, src_nll_loss, src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(0,1))
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_kld_loss, trg_nll_loss, trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(0,1))

        #Variational Inference Loss
        kld_loss = (src_kld_loss + trg_kld_loss)/2
        nll_loss = (src_nll_loss + trg_nll_loss)/2

        #Get the mean of the above losses for each seq and time step
        kld_loss = kld_loss / sample_batched_src['sequence'].shape[0] / sample_batched_src['sequence'].shape[1]
        nll_loss = nll_loss / sample_batched_src['sequence'].shape[0] / sample_batched_src['sequence'].shape[1] / sample_batched_src['sequence'].shape[2] 

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed, None)
        src_domain_loss = F.binary_cross_entropy(src_domain_pred.flatten(), domain_label_src)

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed, None)
        trg_domain_loss =F.binary_cross_entropy(trg_domain_pred.flatten(), domain_label_trg)

        # Total domain loss
        domain_loss = (src_domain_loss + trg_domain_loss)/2

        loss = self.args.weight_KLD*kld_loss + self.args.weight_NLL*nll_loss + src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

        #Losses from variational inference
        self.losses_kld.update(kld_loss.item(), sample_batched_src['sequence'].size(0))
        self.losses_nll.update(nll_loss.item(), sample_batched_src['sequence'].size(0))

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy_score(torch.cat([src_domain_pred, trg_domain_pred]).detach().cpu().numpy().flatten()>0.5, 
                        torch.cat([domain_label_src, domain_label_trg]).detach().cpu().numpy().flatten())
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        self.top1_ts.update(acc1, 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)


    def init_metrics(self):
        self.losses_kld = AverageMeter('Loss KLD', ':.4e')
        self.losses_nll = AverageMeter('Loss NLL', ':.4e')
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_kld, self.losses_nll, self.losses_ts, self.top1_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'domain_classifier_state_dict': self.domain_classifier.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'optimizer_disc' : self.optimizer_disc.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])

    #We need to overwrite predict methods from Base Algorithm (because feature extractor returns more than one element)
    def predict_trg(self, sample_batched):
        _, _, trg_feat = self.feature_extractor(sample_batched['sequence'].transpose(0,1))
        y_pred_trg = self.classifier(trg_feat, sample_batched.get('static'))

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):
        _, _, src_feat = self.feature_extractor(sample_batched['sequence'].transpose(0,1))
        y_pred_src = self.classifier(src_feat, sample_batched.get('static'))

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):
        _, _, feat = self.feature_extractor(sample_batched['sequence'].transpose(0,1))

        return feat

#AdvSKM Algorithm Implementation
class AdvSKM(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(AdvSKM, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)
        self.AdvSKM_embedder = AdvSKM_Disc(input_dim = self.feature_extractor.num_channels[-1], hidden_dim = args.hidden_dim_MLP)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.mmd_loss_calc = MMD_loss(kernel_num=1)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
        self.AdvSKM_embedder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]


        source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
        target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())

        mmd_loss = - self.mmd_loss_calc(source_embedding_disc, target_embedding_disc)

        if self.training:
            # update discriminator if in trainining mode
            mmd_loss.requires_grad = True
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])
        
        # domain loss.
        source_embedding_disc = self.AdvSKM_embedder(src_feat)
        target_embedding_disc = self.AdvSKM_embedder(trg_feat)

        mmd_loss_adv = self.mmd_loss_calc(source_embedding_disc, target_embedding_disc)
       

        loss =  self.args.weight_domain*mmd_loss_adv + src_cls_loss

        if self.training:
            mmd_loss_adv.requires_grad = True
             # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # measure domain loss
        self.losses_ts.update(mmd_loss_adv.item(), sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)


    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'AdvSKM_embedder_state_dict': self.AdvSKM_embedder.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'optimizer_disc' : self.optimizer_disc.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.AdvSKM_embedder.load_state_dict(checkpoint['AdvSKM_embedder_state_dict'])


#CDAN Algorithm Implementation
class CDAN(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(CDAN, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)
        self.domain_classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] * self.output_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = 1, use_batch_norm = args.use_batch_norm)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.criterion_cond = ConditionalEntropyLoss(self.dataset_type, args.task)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
        self.domain_classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        p = float(kwargs.get("count_step")) / 400
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        domain_label_src = torch.ones(len(sample_batched_src['sequence'])).cuda()
        domain_label_trg = torch.zeros(len(sample_batched_trg['sequence'])).cuda()
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]
        trg_pred = self.classifier(trg_feat, sample_batched_trg.get('static'))

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
        disc_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)

        if self.training:
            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(sample_batched_src['sequence'])).cuda()
        domain_label_trg = torch.ones(len(sample_batched_trg['sequence'])).cuda()
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

         # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)), None)
        # loss of domain discriminator according to fake labels
        domain_loss = F.binary_cross_entropy(disc_prediction.flatten(), domain_label_concat)


        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = src_cls_loss + self.args.weight_domain*domain_loss + self.args.weight_cent_target*loss_trg_cent

        # update feature extractor
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy_score(disc_prediction.detach().cpu().numpy().flatten()>0.5, 
                        domain_label_concat.detach().cpu().numpy().flatten())
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        self.top1_ts.update(acc1, 2*sample_batched_src['sequence'].size(0))

        self.lossess_cent.update(loss_trg_cent.item(), sample_batched_trg['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            self.pred_meter_val_trg.update(sample_batched_trg['label'], trg_pred)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.lossess_cent = AverageMeter('Loss CENT', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.top1_ts, self.lossess_cent, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'domain_classifier_state_dict': self.domain_classifier.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'optimizer_disc' : self.optimizer_disc.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])


#DDC Algorithm Implementation
class DDC(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(DDC, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.mmd_loss_calc = MMD_loss(kernel_num=1)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Total domain loss
        domain_loss = self.mmd_loss_calc(src_feat, trg_feat)

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


#CAN Algorithm Implementation
class CAN(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(CAN, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.mmd_loss_calc = MMD_loss(kernel_type="linear", kernel_num=1)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        #Normalize the features
        src_feat_norm = nn.functional.normalize(src_feat, dim=1)
        trg_feat_norm = nn.functional.normalize(trg_feat, dim=1)

        #Get cluster centers from source domain
        cluster_means = []
        #for l in torch.unique(sample_batched_src['label']):
        for l in range(sample_batched_src['label'].max().long()+1):
            if (sample_batched_src['label'].flatten() == l).sum() > 0:
                cluster_mean = src_feat_norm[sample_batched_src['label'].flatten() == l].mean(dim=0)
            #If label doesn't exist in current batch, fill with 999s
            else:
                cluster_mean = torch.Tensor([999]).to(src_feat_norm.device)
                cluster_mean = cluster_mean.repeat_interleave(src_feat_norm.shape[-1])
            cluster_means.append(cluster_mean)
        cluster_means = torch.stack(cluster_means)

        trg_label_pseudo = torch.cdist(trg_feat_norm,cluster_means).argmin(dim=1)

        #Domain loss (MMD) for the same labels
        domain_loss_same = [self.mmd_loss_calc(src_feat_norm[sample_batched_src['label'].flatten() == l], trg_feat_norm[trg_label_pseudo == l]) for l in torch.unique(trg_label_pseudo)]
        domain_loss_same = torch.stack(domain_loss_same).mean()

        #Domain loss (MMD) across different labels
        domain_loss_diff = [self.mmd_loss_calc(src_feat_norm[sample_batched_src['label'].flatten() == ls], trg_feat_norm[trg_label_pseudo == lt]) for lt in torch.unique(trg_label_pseudo) for ls in torch.unique(sample_batched_src['label'])]
        domain_loss_diff = torch.stack(domain_loss_diff).mean()

        # Total domain loss
        domain_loss = domain_loss_same - domain_loss_diff

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])



#DeepCORAL Algorithm Implementation
class DeepCORAL(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(DeepCORAL, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.coral = CORAL()

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Total domain loss
        domain_loss = self.coral(src_feat, trg_feat)

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


#DSAN Algorithm Implementation
class DSAN(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(DSAN, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        #For binary classification (i.e. output_dim=1), we actually have 2 classes
        self.lmmd_loss_calc = LMMD_loss(class_num = 2 if self.output_dim == 1 else self.output_dim , kernel_num=1)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]
        trg_pred = self.classifier(trg_feat, sample_batched_trg.get('static'))

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)


        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Total domain loss
        domain_loss = self.lmmd_loss_calc.get_loss(src_feat, trg_feat, sample_batched_src['label'], self.get_trg_pred_lmmd(trg_pred))

        # total loss
        loss = src_cls_loss + self.args.weight_domain*domain_loss 

        # update feature extractor
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            self.pred_meter_val_trg.update(sample_batched_trg['label'], trg_pred)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])

    def get_trg_pred_lmmd(self, trg_pred):
        #For other tasks (binary), we output the probability of class 1. 
        #that's why we append the prob of class 0 as another column to the beginning
        if self.dataset_type == "icu" and self.args.task != "los":
            return torch.cat([1-trg_pred,trg_pred],dim=-1)
        #If the task is multiclass classification, our models output unnormalized scores for each class/category. 
        #That's why we get softmax
        else:
            return torch.nn.functional.softmax(trg_pred, dim=1)


#HoMM Algorithm Implementation
class HoMM(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(HoMM, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.homm_loss = HoMM_loss()

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Total domain loss
        domain_loss = self.homm_loss(src_feat, trg_feat)

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts.update(domain_loss.item(), 2*sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_ts = AverageMeter('Loss Sour-Tar', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


#MMDA Algorithm Implementation
class MMDA(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(MMDA, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.coral = CORAL()
        self.criterion_cond = ConditionalEntropyLoss(self.dataset_type, args.task)
        self.mmd_loss_calc = MMD_loss(kernel_num=1)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]
        trg_pred = self.classifier(trg_feat, sample_batched_trg.get('static'))

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        # Total domain loss
        loss_coral = self.coral(src_feat, trg_feat)
        loss_cent = self.criterion_cond(trg_pred)
        loss_mmd = self.mmd_loss_calc(src_feat, trg_feat)
        domain_loss = loss_coral + loss_cent + loss_mmd

        loss = src_cls_loss + self.args.weight_domain*domain_loss

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        self.losses_ts_coral.update(loss_coral.item(), sample_batched_src['sequence'].size(0))
        self.losses_ts_cent.update(loss_cent.item(), sample_batched_src['sequence'].size(0))
        self.losses_ts_mmd.update(loss_mmd.item(), sample_batched_src['sequence'].size(0))
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            self.pred_meter_val_trg.update(sample_batched_trg['label'], trg_pred)

    def init_metrics(self):
        self.losses_ts_coral = AverageMeter('Loss Sour-Tar CORAL', ':.4e')
        self.losses_ts_cent = AverageMeter('Loss Sour-Tar CENT', ':.4e')
        self.losses_ts_mmd = AverageMeter('Loss Sour-Tar MMD', ':.4e')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_ts_coral, self.losses_ts_cent, self.losses_ts_mmd, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


#TCN (i.e. w/o UDA) Algorithm Implementation
class TCN(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(TCN, self).__init__(args)

        self.feature_extractor = self.build_feature_extractor_TCN(args, input_channels_dim, self.num_channels)
        self.classifier = MLP(input_dim = self.feature_extractor.num_channels[-1] + input_static_dim, hidden_dim = args.hidden_dim_MLP, 
                                   output_dim = self.output_dim, use_batch_norm = args.use_batch_norm)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):

        src_feat = self.feature_extractor(sample_batched_src['sequence'].transpose(1,2))[:,:,-1]
        src_pred = self.classifier(src_feat, sample_batched_src.get('static'))

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(src_pred, sample_batched_src['label'])

        loss = src_cls_loss 

        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        
        self.losses_pred.update(src_cls_loss.item(), sample_batched_src['sequence'].size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], src_pred)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], src_pred)

            #keep track of prediction results (of target) explicitly
            trg_feat = self.feature_extractor(sample_batched_trg['sequence'].transpose(1,2))[:,:,-1]
            y_pred_trg = self.classifier(trg_feat, sample_batched_trg.get('static'))

            self.pred_meter_val_trg.update(sample_batched_trg['label'], y_pred_trg)

    def init_metrics(self):
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'feature_extractor_state_dict': self.feature_extractor.state_dict(),
                        'classifier_state_dict': self.classifier.state_dict(),
                        'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.classifier.load_state_dict(checkpoint['classifier_state_dict'])


#CLUDA Algorithm Implementation
class CLUDA(Base_Algorithm):

    def __init__(self, args, input_channels_dim, input_static_dim):

        super(CLUDA, self).__init__(args)

        self.input_channels_dim = input_channels_dim
        self.input_static_dim = input_static_dim

        #different from other algorithms, we import entire model at onces. (i.e. no separate feature extractor or classifier)
        self.model = DA_MoCoNNQQ_Disc_TCN_Siam(num_inputs=(1+args.use_mask)*input_channels_dim, output_dim=self.output_dim, num_channels=self.num_channels, num_static=input_static_dim, 
            mlp_hidden_dim=args.hidden_dim_MLP, use_batch_norm=args.use_batch_norm, kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout, K=args.queue_size, m=args.momentum)

        self.augmenter = None
        self.concat_mask = concat_mask

        self.criterion_CL = nn.CrossEntropyLoss()

        self.cuda()

        self.optimizer = torch.optim.Adam(
        self.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, betas=(0.5, 0.99)
        )

        self.init_metrics()


    def step(self, sample_batched_src, sample_batched_trg, **kwargs):
        #For Augmenter, Cutout length is calculated relative to the sequence length
        #If there is only one channel, there will be no spatial dropout
        if self.augmenter is None:
            self.get_augmenter(sample_batched_src)

        p = float(kwargs.get("count_step")) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        #Go through augmentations first
        seq_q_src, seq_mask_q_src = self.augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])
        seq_k_src, seq_mask_k_src = self.augmenter(sample_batched_src['sequence'], sample_batched_src['sequence_mask'])
        
        seq_q_trg, seq_mask_q_trg = self.augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])
        seq_k_trg, seq_mask_k_trg = self.augmenter(sample_batched_trg['sequence'], sample_batched_trg['sequence_mask'])

        #Concat mask if use_mask = True
        seq_q_src = self.concat_mask(seq_q_src, seq_mask_q_src, self.args.use_mask)
        seq_k_src = self.concat_mask(seq_k_src, seq_mask_k_src, self.args.use_mask)
        seq_q_trg = self.concat_mask(seq_q_trg, seq_mask_q_trg, self.args.use_mask)
        seq_k_trg = self.concat_mask(seq_k_trg, seq_mask_k_trg, self.args.use_mask)

        # compute output
        output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s = self.model(seq_q_src, seq_k_src, sample_batched_src.get('static'), seq_q_trg, seq_k_trg, sample_batched_trg.get('static'), alpha)

        # Compute all losses
        loss_s = self.criterion_CL(output_s, target_s)
        loss_t = self.criterion_CL(output_t, target_t)
        loss_ts = self.criterion_CL(output_ts, target_ts)
        loss_disc = F.binary_cross_entropy(output_disc, target_disc)

        # Task classification  Loss
        src_cls_loss = self.pred_loss.get_prediction_loss(pred_s, sample_batched_src['label'])

        loss = self.args.weight_loss_src*loss_s + self.args.weight_loss_trg*loss_t + \
                    self.args.weight_loss_ts*loss_ts + self.args.weight_loss_disc*loss_disc + self.args.weight_loss_pred*src_cls_loss


        #If in training mode, do the backprop
        if self.training:

            # zero grad
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1 = accuracy(output_s, target_s, topk=(1, ))
        self.losses_s.update(loss_s.item(), seq_q_src.size(0))
        self.top1_s.update(acc1[0][0], seq_q_src.size(0))
        
        acc1 = accuracy(output_t, target_t, topk=(1, ))
        self.losses_t.update(loss_t.item(), seq_q_trg.size(0))
        self.top1_t.update(acc1[0][0], seq_q_trg.size(0))

        acc1 = accuracy(output_ts, target_ts, topk=(1, ))
        self.losses_ts.update(loss_t.item(), seq_q_trg.size(0))
        self.top1_ts.update(acc1[0][0], seq_q_trg.size(0))
        
        acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten()>0.5, target_disc.detach().cpu().numpy().flatten())
        self.losses_disc.update(loss_disc.item(), output_disc.size(0))
        self.top1_disc.update(acc1, output_disc.size(0))
        
        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], pred_s)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], pred_s)

            #keep track of prediction results (of target) explicitly
            pred_t = self.model.predict(seq_q_trg, sample_batched_trg.get('static'), is_target=True)

            self.pred_meter_val_trg.update(sample_batched_trg['label'], pred_t)

    def init_metrics(self):

        self.losses_s = AverageMeter('Loss Source', ':.4e')
        self.top1_s = AverageMeter('Acc@1', ':6.2f')
        self.losses_t = AverageMeter('Loss Target', ':.4e')
        self.top1_t = AverageMeter('Acc@1', ':6.2f')
        self.losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_disc = AverageMeter('Loss Sour-Tar Disc', ':.4e')
        self.top1_disc = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_s, self.top1_s, self.losses_t, self.top1_t, self.losses_ts, self.top1_ts, 
                self.losses_disc,  self.top1_disc, self.losses_pred, self.score_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    #We need to overwrite below functions for CLUDA
    def predict_trg(self, sample_batched):

        seq_t = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_trg = self.model.predict(seq_t, sample_batched.get('static'), is_target=True)

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):

        seq_s = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_src = self.model.predict(seq_s, sample_batched.get('static'), is_target=False)

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):

        seq = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        feat = self.model.get_encoding(seq)

        return feat

    def get_augmenter(self, sample_batched):

        seq_len = sample_batched['sequence'].shape[1]
        num_channel = sample_batched['sequence'].shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        #IF THERE IS ONLY ONE CHANNEL, WE NEED TO MAKE SURE THAT CUTOUT AND CROPOUT APPLIED (i.e. their probs are 1)
        #for extremely long sequences (such as SSC with 3000 time steps)
        #apply the cutout in multiple places, in return, reduce history crop
        elif self.input_channels_dim == 1 and seq_len>1000: 
            self.augmenter = Augmenter(cutout_length=cutout_len, cutout_prob=1, crop_min_history=0.25, crop_prob=1, dropout_prob=0.0)
            #we apply cutout 3 times in a row.
            self.augmenter.augmentations = [self.augmenter.history_cutout, self.augmenter.history_cutout, self.augmenter.history_cutout,
                                            self.augmenter.history_crop, self.augmenter.gaussian_noise, self.augmenter.spatial_dropout]
        #if there is only one channel but not long, we just need to make sure that we don't drop this only channel
        else:
            self.augmenter = Augmenter(cutout_length=cutout_len, dropout_prob=0.0)


