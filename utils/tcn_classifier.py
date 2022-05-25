import torch
import torch.nn as nn

from .tcn_no_norm import TemporalConvNet
from .mlp import MLP

class TCN_Classifier(nn.Module):
    def __init__(self, num_inputs, output_dim, num_channels, num_static, mlp_hidden_dim=256, use_batch_norm=False, use_mask=False, kernel_size=2, dropout=0.2):
        super(TCN_Classifier, self).__init__()
        self.tcn  = TemporalConvNet((1+use_mask)*num_inputs, num_channels, kernel_size, dropout)
        #self.linear = nn.Linear(num_channels[-1] + num_static, 1)
        #self.sigmoid = nn.Sigmoid()
        self.mlp = MLP(num_channels[-1] + num_static, mlp_hidden_dim, output_dim=output_dim, use_batch_norm = use_batch_norm)
        
        self.use_mask = use_mask

    def forward(self, sequence, sequence_mask, static):
        #if the mask is used, it is concatenated as new channels to the sequence
        if self.use_mask:
            sequence = torch.cat([sequence, sequence_mask], dim=2)
        
        #Input is in the shape N*L*C whereas TCN expects N*C*L
        
        tcn_out = self.tcn(sequence.transpose(1,2))
        
        #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
        tcn_out = tcn_out[:,:,-1]
        #Concatenate the tcn_output and static features
        #out = torch.cat((tcn_out, static), dim=-1)
        #out = self.linear(out)
        #out = self.sigmoid(out)
        
        out = self.mlp(tcn_out, static)
        return out

    def get_encoding(self, sequence):
        #if the mask is used, it is concatenated as new channels to the sequence
        if self.use_mask:
            sequence = torch.cat([sequence, sequence_mask], dim=2)
        
        #Input is in the shape N*L*C whereas TCN expects N*C*L
        
        tcn_out = self.tcn(sequence.transpose(1,2))
        
        #Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output 
        tcn_out = tcn_out[:,:,-1]

        return tcn_out