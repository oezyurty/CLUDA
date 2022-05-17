import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class ICUDataset(Dataset):
    
    def __init__(self, root_dir, split_type="train", stay_hours=48, min_hours=4, task="mortality", is_full_subset=False, subsample_patients=1.0, is_cuda=True, verbose=False):
        """
        Inputs:
            root_dir: Directory to load ICU stays
            split_type: One of {"train", "val", "test"}. If test, all sub-sequences (>= min_hours) will be tested.
            stay_hours: Length of each sequence. If sequence is shorter than stay_hours, it will be padded; else, it will be randomly cropped
            min_hours: Sequences shorter than min_hours will be ignored (i.e. removed when loading the data)
            task: It determines the label. One of {"mortality", "decompensation", "los"}:
                - mortality: Binary outcome of the ICU stay.
                - decompensation: Binary outcome for the mortality within the next 24 hours. 
                - los: Remaining length of stay (in hours) in ICU -- NOTE: For prediction, it is binned as [(0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6) (6, 7) (7, 8) (8, 14) (14+)] days.
            is_full_subset: Flag to yield all subsequences of an ICU stay longer than 'min_hours' with max length 'stay_hours'.
            subsample_patients: Ratio of patients to be subsampled from dataset. Default is keeping all the patients. (Useful for Domain Adaptation Setting.)
            is_cuda: Flag for sending the tensors to gpu or not
            verbose: Flag for printing out some internal operations (for debugging)
            
        """
        self.root_dir = root_dir
        self.split_type = split_type
        self.stay_hours = stay_hours
        self.min_hours = min_hours
        self.task = task
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.is_full_subset = is_full_subset
        self.subsample_patients = subsample_patients
        
        self.load_stays()
        
    def __len__(self):
        if not self.is_full_subset:
            return len(self.mortality)
        else:
            return len(self.stay_dict)
        
        
    def __getitem__(self, id_):

        #For a given id, randomly return a time window per patient (under certain conditions)
        if not self.is_full_subset:
            idx = id_
        
            sequence = self.sequence[idx]
            sequence_mask = self.sequence_mask[idx]
            static = self.static[idx]
            mortality = self.mortality[idx]

            #If it is train or val, we provide one random time window per patient (under certain conditions)
            if not self.split_type == "test": 

                #Get the last stay_hours, starting from min_hours
                end_index = np.random.randint(self.min_hours, len(sequence)+1) #get a random number between [min_hours+1, length]

                sequence, sequence_mask, label = self.get_subsequence(sequence, sequence_mask, mortality, end_index)
                
                if self.is_cuda:
                    sequence = torch.Tensor(sequence).float().cuda()
                    sequence_mask = torch.Tensor(sequence_mask).long().cuda()
                    static = torch.Tensor(static).float().cuda()
                    label = torch.Tensor([label]).float().cuda() if self.task != "los" else torch.Tensor([label]).long().cuda()

                else:
                    sequence = torch.Tensor(sequence).float()
                    sequence_mask = torch.Tensor(sequence_mask).long()
                    static = torch.Tensor(static).float()
                    label = torch.Tensor([label]).float() if self.task != "los" else torch.Tensor([label]).long().cuda()
                
                    
                sample = {"sequence":sequence, "sequence_mask":sequence_mask, "static":static, "label":label, "patient_id":idx, "stay_hour":end_index}

            #if it is test, we provide ALL subsequences per patient (starting from min_hours)

        else:
            idx, end_index = self.stay_dict[id_]

            sequence = self.sequence[idx]
            sequence_mask = self.sequence_mask[idx]
            static = self.static[idx]
            mortality = self.mortality[idx]

            sequence, sequence_mask, label = self.get_subsequence(sequence, sequence_mask, mortality, end_index)
                
            if self.is_cuda:
                sequence = torch.Tensor(sequence).float().cuda()
                sequence_mask = torch.Tensor(sequence_mask).long().cuda()
                static = torch.Tensor(static).float().cuda()
                label = torch.Tensor([label]).float().cuda() if self.task != "los" else torch.Tensor([label]).long().cuda()
            else:
                sequence = torch.Tensor(sequence).float()
                sequence_mask = torch.Tensor(sequence_mask).long()
                static = torch.Tensor(static).float()
                label = torch.Tensor([label]).float() if self.task != "los" else torch.Tensor([label]).long().cuda()
            
                
            sample = {"sequence":sequence, "sequence_mask":sequence_mask, "static":static, "label":label, "patient_id":idx, "stay_hour":end_index}
        
        return sample
        
    
    def load_stays(self):
        path_sequence = os.path.join(self.root_dir, "timeseries_"+self.split_type+".npy")
        path_sequence_mask = os.path.join(self.root_dir, "timeseries_"+self.split_type+"_mask.npy")
        path_static = os.path.join(self.root_dir, "static_"+self.split_type+".npy")
        path_mortality = os.path.join(self.root_dir, "mortality_"+self.split_type+".npy")
        
        self.sequence = np.load(path_sequence, allow_pickle=True)
        self.sequence_mask = np.load(path_sequence_mask, allow_pickle=True)
        self.static = np.load(path_static, allow_pickle=True)
        self.mortality = np.load(path_mortality, allow_pickle=True)

        #Filter out the stays shorter than min_hours
        seq_lengths = np.array(list(map(lambda x: len(x), self.sequence)))
        filter_cond = seq_lengths >= self.min_hours

        if self.verbose:
            num_removed = (1 - filter_cond).sum()
            print ("Number of sequences removed: " + str(num_removed))


        self.sequence = self.sequence[filter_cond]
        self.sequence_mask = self.sequence_mask[filter_cond]
        self.static = self.static[filter_cond]
        self.mortality = self.mortality[filter_cond]

        if self.subsample_patients < 1.0:
            cond_subsample = np.random.choice(a=[True, False], size=(len(self.sequence)), p=[self.subsample_patients, 1-self.subsample_patients])
            num_removed = (1 - cond_subsample).sum()
            print ("Number of sequences removed for subsampling: " + str(num_removed))

            self.sequence = self.sequence[cond_subsample]
            self.sequence_mask = self.sequence_mask[cond_subsample]
            self.static = self.static[cond_subsample]
            self.mortality = self.mortality[cond_subsample]

        #To reduce memory consumption, we will only keep the necessary elements as a list (id and end_index)
        # to generate the data when needed
        if self.is_full_subset:

            self.stay_dict = np.concatenate(list(map(lambda seq, i: self.seq_to_dict_helper(seq, i), self.sequence, np.arange(len(self.sequence)))), axis=0)



    def seq_to_dict_helper(self, sequence, id_):
        end_indices = np.arange(self.min_hours, len(sequence)+1).reshape(-1,1)
        ids = np.repeat(id_, len(end_indices)).reshape(-1,1)

        return np.concatenate([ids, end_indices], axis=1)


    def get_subsequence(self, sequence, sequence_mask, mortality, end_index):
        """
        Given the end index (i.e. last time step considered)
        this function returns the padded/cropped sequence and its corresponding label
        """
        if self.task == "mortality":
            label = mortality
        elif self.task == "decompensation":
            #If ICU mortality is 1, and last time step is within the last 24 hours of ICU stay, decompensation is 1.
            label = 1 if mortality == 1 and end_index+24>len(sequence) else 0
        else:
            #If the task is length of stay
            los_hours = len(sequence) - end_index
            label = self.get_los_bin(los_hours)

        #Update the sequence based on the end_index
        #if chosen end_index is smaller than stay_hours, we will need pre-padding
        if end_index < self.stay_hours:
            sequence = sequence[:end_index, : ]
            sequence_mask = sequence[:end_index, : ]
            pad_len = self.stay_hours - len(sequence)
            sequence = np.pad(sequence, ((pad_len,0),(0,0)), 'constant', constant_values=((0,0),(0,0)))
            sequence_mask = np.pad(sequence_mask, ((pad_len,0),(0,0)), 'constant', constant_values=((0,0),(0,0)))
        else:
            sequence = sequence[end_index - self.stay_hours:end_index, :]
            sequence_mask = sequence_mask[end_index - self.stay_hours:end_index, :]

        return sequence, sequence_mask, label

    #Convert LOS(hours) to bin (10 categories) as it was described in previous benchmarks.
    def get_los_bin(self, los):
        #Bins are (0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6) (6, 7) (7, 8) (8, 14) (14+) in days. 
        if los < 24*8:
            label = los // 24
        elif los >= 24*8 and los<24*14:
            label = 8
        else:
            label = 9
        return label

def collate_test(batch):
    #The input is list of dictionaries
    out = {}
    for key in batch[0].keys():
        val = []
        for sample in batch:
            val.append(sample[key])
        val = torch.cat(val, dim=0)
        out[key] = val
    return out


