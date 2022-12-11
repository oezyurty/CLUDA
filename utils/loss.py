import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLoss(object):
    def __init__(self, dataset_type="icu", task="decompensation", weight_ratio=None):
        self.dataset_type = dataset_type
        self.task = task
        self.weight_ratio = weight_ratio
        self.get_loss_weights()

    #Calculate weights for binary classification, such that average will be around 1,
    # yet minority class will have "weight_ratio" times more weight.
    def get_loss_weights(self):
            prop = 1 - (1/(self.weight_ratio + 1))
            weight_0 = 1 / (2 * prop)
            weight_1 = weight_0 * prop / (1 - prop)
            self.loss_weights = (weight_0, weight_1)

    #Return the appropriate prediction loss for the related task
    def get_prediction_loss(self, output, target):
        if self.dataset_type == "icu" and self.task != "los":
            batch_loss_weights = target * self.loss_weights[1] + (1 - target) * self.loss_weights[0]
            loss = F.binary_cross_entropy(output, target, weight = batch_loss_weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
            #Currently, target has the shape (N,1), we need to flatten
            loss = loss_fn(output, target.squeeze(1))
        return loss
    
