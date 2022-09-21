#Time Series Version of MoCo, with TCN used as encoder 
import sys
sys.path.append("../..")
import torch
import torch.nn as nn
import numpy as np

#Import our encoder 
from utils.tcn_no_norm import TemporalConvNet
from utils.nearest_neighbor import NN, sim_matrix
from utils.mlp import MLP


#Helper function for reversing the discriminator backprop
from torch.autograd import Function
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DA_MoCoNNQQ_Disc_TCN_Siam(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, num_inputs, output_dim, num_channels, num_static, mlp_hidden_dim=256, use_batch_norm=True, num_neighbors = 1, kernel_size=2, stride=1, dilation_factor=2, dropout=0.2, K=24576, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DA_MoCoNNQQ_Disc_TCN_Siam, self).__init__()

        self.sigmoid = nn.Sigmoid()

        self.K = K
        self.m = m
        self.T = T
        self.num_neighbors = num_neighbors

        # encoders
        # num_classes is the output fc dimension
        self.encoder_q = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, 
                                            stride=stride, dilation_factor=dilation_factor, dropout=dropout)
        self.encoder_k = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size, 
                                            stride=stride, dilation_factor=dilation_factor, dropout=dropout)

        #projector for query
        self.projector = MLP(input_dim = num_channels[-1] , hidden_dim = mlp_hidden_dim, 
                               output_dim = num_channels[-1], use_batch_norm = use_batch_norm)

        #Classifier trained by source query
        self.predictor = MLP(input_dim = num_channels[-1] + num_static, hidden_dim = mlp_hidden_dim, 
                               output_dim = output_dim, use_batch_norm = use_batch_norm)

        #Discriminator
        self.discriminator = MLP(input_dim = num_channels[-1], hidden_dim = mlp_hidden_dim, 
                               output_dim = 1, use_batch_norm = use_batch_norm)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_s", torch.randn(num_channels[-1], K))
        self.queue_s = nn.functional.normalize(self.queue_s, dim=0)

        self.register_buffer("queue_t", torch.randn(num_channels[-1], K))
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        #No update during evaluation
        if self.training:
            #Update the encoder
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        #No update during evaluation
        if self.training:
            # gather keys before updating queue
            batch_size = keys_s.shape[0]

            ptr = int(self.queue_ptr)
            #For now, ignore below assertion
            #assert self.K % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue_s[:, ptr:ptr + batch_size] = keys_s.T
            self.queue_t[:, ptr:ptr + batch_size] = keys_t.T

            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr

    def forward(self, sequence_q_s, sequence_k_s, static_s, sequence_q_t, sequence_k_t, static_t, alpha):
        """
        Input:
            sequence_q: a batch of query sequences
            sequence_k: a batch of key sequences
            static: a batch of static features
        Output:
            logits, targets
        """

        #SOURCE DATASET query computations

        # compute query features
        #Input is in the shape N*L*C whereas TCN expects N*C*L
        #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
        q_s = self.encoder_q(sequence_q_s.transpose(1,2))[:,:,-1]  # queries: NxC
        q_s = nn.functional.normalize(q_s, dim=1)
        #Project the query
        p_q_s = self.projector(q_s, None)  # queries: NxC
        p_q_s = nn.functional.normalize(p_q_s, dim=1)

        #TARGET DATASET query computations

        # compute query features
        #Input is in the shape N*L*C whereas TCN expects N*C*L
        #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
        q_t = self.encoder_q(sequence_q_t.transpose(1,2))[:,:,-1]  # queries: NxC
        q_t = nn.functional.normalize(q_t, dim=1)
        #Project the query
        p_q_t = self.projector(q_t, None)  # queries: NxC
        p_q_t = nn.functional.normalize(p_q_t, dim=1)



        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoders

            #SOURCE DATASET key computations

            #Input is in the shape N*L*C whereas TCN expects N*C*L
            #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
            k_s = self.encoder_k(sequence_k_s.transpose(1,2))[:,:,-1]  # queries: NxC
            k_s = nn.functional.normalize(k_s, dim=1)

            #TARGET DATASET key computations

            #Input is in the shape N*L*C whereas TCN expects N*C*L
            #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
            k_t = self.encoder_k(sequence_k_t.transpose(1,2))[:,:,-1]  # queries: NxC
            k_t = nn.functional.normalize(k_t, dim=1)


        
        #SOURCE DATASET contrastive loss
        #Calculate the logits of the given batch: NxN
        l_batch_s = torch.mm(p_q_s, k_s.transpose(0,1))
        #Calculate the logits of the queue: NxK
        l_queue_s = torch.mm(p_q_s, self.queue_s.clone().detach())
        
        # logits Nx(N+K)
        logits_s = torch.cat([l_batch_s, l_queue_s], dim=1)

        # apply temperature
        logits_s /= self.T

        #labels
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long).to(device = p_q_s.device)


        #TARGET DATASET contrastive loss
        #Calculate the logits of the given batch: NxN
        l_batch_t = torch.mm(p_q_t, k_t.transpose(0,1))
        #Calculate the logits of the queue: NxK
        l_queue_t = torch.mm(p_q_t, self.queue_t.clone().detach())
        
        # logits Nx(N+K)
        logits_t = torch.cat([l_batch_t, l_queue_t], dim=1)

        # apply temperature
        logits_t /= self.T

        #labels
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long).to(device = p_q_t.device)

        # TARGET-SOURCE Contrastive loss: 
        # We want the target query (not its projection!) to get closer to its key's NN in source query.

        _, indices_nn = NN(k_t, q_s.clone().detach(), num_neighbors = self.num_neighbors, return_indices=True)

        #logits for NNs: NxN
        logits_ts = torch.mm(q_t, q_s.transpose(0,1).clone().detach())
        
        # apply temperature
        logits_ts /= self.T
        
        #labels
        labels_ts = indices_nn.squeeze(1).to(device = q_t.device)

        # DOMAIN DISCRIMINATION Loss
        
        domain_label_s = torch.ones((len(q_s),1)).to(device = q_s.device)
        domain_label_t = torch.zeros((len(q_t),1)).to(device = q_t.device)

        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)

        q_s_reversed = ReverseLayerF.apply(q_s, alpha)
        q_t_reversed = ReverseLayerF.apply(q_t, alpha)

        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        pred_domain = self.discriminator(q_reversed, None)

        #SOURCE Prediction task
        y_s = self.predictor(q_s, static_s)

        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k_s, k_t)

        return logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts, pred_domain, labels_domain, y_s
    
    def get_encoding(self, sequence, is_target=True):
        # compute the encoding of a sequence (i.e. before projection layer)
        #Input is in the shape N*L*C whereas TCN expects N*C*L
        #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
        
        #We will use the encoder from a given domain (either source or target)
        
        q = self.encoder_q(sequence.transpose(1,2))[:,:,-1]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        
        return q
    
    def predict(self, sequence, static, is_target=True):
        #Get the encoding of a sequence from a given domain    
        q = self.get_encoding(sequence, is_target=is_target)
        
        #Make the prediction based on the encoding
        y = self.predictor(q, static)
        
        return y