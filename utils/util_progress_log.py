import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, cohen_kappa_score
import logging

def get_logger(log_file):
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    return log



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, is_logged=False):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        #If the progress will be logged (instead of printing)
        #we will return the string so that it can be logged in the main script.
        if not is_logged:
            print('\t'.join(entries))
        else:
            return '\t'.join(entries)


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class PredictionMeter(object):
    def __init__(self, task="decompensation"):
        self.task = task
        self.target_list = []
        self.output_list = []
        self.id_patient_list = []
        self.stay_hours_list = []

    def update(self, target, output, id_patient = None, stay_hour = None):
        if self.task != "los":
            output_np = output.detach().cpu().numpy().flatten()
        else:
            output_np = output.detach().cpu().numpy().argmax(axis=1).flatten()
        target_np = target.detach().cpu().numpy().flatten()

        self.output_list = self.output_list + list(output_np)
        self.target_list = self.target_list + list(target_np)

        #Below is especially helpful for saving the predictions in eval mode
        if id_patient is not None:
            id_patient_np = id_patient.numpy().flatten()
            self.id_patient_list = self.id_patient_list + list(id_patient_np)

        if stay_hour is not None:
            stay_hour_np = stay_hour.numpy().flatten()
            self.stay_hours_list = self.stay_hours_list + list(stay_hour_np)

    def get_metrics(self):
        return_dict = {}
        output = np.array(self.output_list)
        target = np.array(self.target_list)
        if self.task != "los":
            roc_auc = roc_auc_score(target, output)
            avg_prc = average_precision_score(target, output)

            return_dict["roc_auc"] = roc_auc
            return_dict["avg_prc"] = avg_prc

        else:
            kappa = cohen_kappa_score(output, target)
            return_dict["kappa"] = kappa

        return return_dict

    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def write_to_tensorboard(writer, progress, count_step, split_type="train", task="decompensation", auroc_s=None, auprc_s=None, auroc_t=None, auprc_t=None, kappa_s=None, kappa_t=None):
    if len(progress.meters) == 11:
        writer.add_scalar('Loss_Total/'+split_type, progress.meters[10].avg, count_step)
        writer.add_scalar('Loss_Source_CL/'+split_type, progress.meters[2].avg, count_step)
        writer.add_scalar('Loss_Target_CL/'+split_type, progress.meters[4].avg, count_step)
        writer.add_scalar('Loss_SourTar_CL/'+split_type, progress.meters[6].avg, count_step)
        writer.add_scalar('Loss_Source_Pred/'+split_type, progress.meters[8].avg, count_step)

        writer.add_scalar('Acc@1_Source_CL/'+split_type, progress.meters[3].avg, count_step)
        writer.add_scalar('Acc@1_Target_CL/'+split_type, progress.meters[5].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_CL/'+split_type, progress.meters[7].avg, count_step)

        if task != "los":
            if split_type == "train":
                writer.add_scalar('AUROC_Source_Pred/'+split_type, progress.meters[9].avg, count_step)
            
            if split_type == "val":
                writer.add_scalar('AUROC_Source_Pred/'+split_type, auroc_s, count_step)
                writer.add_scalar('AUPRC_Source_Pred/'+split_type, auprc_s, count_step)
                writer.add_scalar('AUROC_Target_Pred/'+split_type, auroc_t, count_step)
                writer.add_scalar('AUPRC_Target_Pred/'+split_type, auprc_t, count_step)

        else:
            if split_type == "train":
                writer.add_scalar('KAPPA_Source_Pred/'+split_type, progress.meters[9].avg, count_step)
            
            if split_type == "val":
                writer.add_scalar('KAPPA_Source_Pred/'+split_type, kappa_s, count_step)
                writer.add_scalar('KAPPA_Target_Pred/'+split_type, kappa_t, count_step)

    elif len(progress.meters) == 13:
        writer.add_scalar('Loss_Total/'+split_type, progress.meters[12].avg, count_step)
        writer.add_scalar('Loss_Source_CL/'+split_type, progress.meters[2].avg, count_step)
        writer.add_scalar('Loss_Target_CL/'+split_type, progress.meters[4].avg, count_step)
        writer.add_scalar('Loss_SourTar_CL/'+split_type, progress.meters[6].avg, count_step)
        writer.add_scalar('Loss_SourTar_Disc/'+split_type, progress.meters[8].avg, count_step)
        writer.add_scalar('Loss_Source_Pred/'+split_type, progress.meters[10].avg, count_step)

        writer.add_scalar('Acc@1_Source_CL/'+split_type, progress.meters[3].avg, count_step)
        writer.add_scalar('Acc@1_Target_CL/'+split_type, progress.meters[5].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_CL/'+split_type, progress.meters[7].avg, count_step)
        writer.add_scalar('Acc@1_SourTar_Disc/'+split_type, progress.meters[7].avg, count_step)

        if task != "los":
            if split_type == "train":
                writer.add_scalar('AUROC_Source_Pred/'+split_type, progress.meters[11].avg, count_step)
            
            if split_type == "val":
                writer.add_scalar('AUROC_Source_Pred/'+split_type, auroc_s, count_step)
                writer.add_scalar('AUPRC_Source_Pred/'+split_type, auprc_s, count_step)
                writer.add_scalar('AUROC_Target_Pred/'+split_type, auroc_t, count_step)
                writer.add_scalar('AUPRC_Target_Pred/'+split_type, auprc_t, count_step)

        else:
            if split_type == "train":
                writer.add_scalar('KAPPA_Source_Pred/'+split_type, progress.meters[11].avg, count_step)
            
            if split_type == "val":
                writer.add_scalar('KAPPA_Source_Pred/'+split_type, kappa_s, count_step)
                writer.add_scalar('KAPPA_Target_Pred/'+split_type, kappa_t, count_step)


