import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, use_batch_norm=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None

    def forward(self, x, static):
        
        if static is not None:
            hidden = self.input_fc(torch.cat([x, static], dim=1))
        else:
            hidden = self.input_fc(x)
        
        if self.use_batch_norm:
            hidden = self.batch_norm(hidden)

        hidden = F.relu(hidden)
        
        y_pred = self.output_fc(hidden)
        
        if self.output_dim == 1:
            y_pred = self.sigmoid(y_pred)

        return y_pred