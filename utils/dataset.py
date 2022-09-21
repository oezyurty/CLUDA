import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def get_dataset(args, domain_type, split_type):
    """
    Return the correct dataset object that will be fed into datalaoder
    args: args of main script
    domain_type: "source" or "target"
    split_type: "train" or "val" or "test"

    Note: If args.path_src (or trg) includes "miiv" or "aumc", it will return ICUDataset
    Otherwise, it will return SensorDataset
    """
    
    if "miiv" in args.path_src or "aumc" in args.path_src:
        if domain_type == "source":
            return ICUDataset(args.path_src, task=args.task, split_type=split_type, age_group=getattr(args, "age_src", -1), is_full_subset=True, is_cuda=True)
        else:
            return ICUDataset(args.path_trg, task=args.task, split_type=split_type, age_group=getattr(args, "age_trg", -1), is_full_subset=True, is_cuda=True)

    else:
        if domain_type == "source":
            return SensorDataset(args.path_src, subject_id=args.id_src, split_type=split_type, is_cuda=True)
        else:
            return SensorDataset(args.path_trg, subject_id=args.id_trg, split_type=split_type, is_cuda=True)


def get_output_dim(args):
    """
    It is hard-coded output dims for each dataset and task
    FOR ICU datasets: output dim is 1 for mortality and decompensation, and 10 for los
    For Sensor Datasets: output dim is 6 for WISDM,HAR and HHAR, and 5 for SSC
    """
    output_dim = -1

    if "miiv" in args.path_src or "aumc" in args.path_src:
        if args.task != "los":
            output_dim = 1
        else:
            output_dim = 10
    elif "SSC" in args.path_src:
        output_dim = 5
    else:
        output_dim = 6

    return output_dim


class ICUDataset(Dataset):
    
    def __init__(self, root_dir, split_type="train", age_group=-1, stay_hours=48, min_hours=4, task="mortality", is_full_subset=False, subsample_patients=1.0, is_cuda=True, verbose=False):
        """
        Inputs:
            root_dir: Directory to load ICU stays
            split_type: One of {"train", "val", "test"}. If test, all sub-sequences (>= min_hours) will be tested.
            age_group: There are 5 age groups: 1-> 0-19, 2-> 20-45, 3-> 46-65, 4-> 66-85, 5->85+ 
                If -1, all age groups will be used. 
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
        self.age_group = age_group
        self.stay_hours = stay_hours
        self.min_hours = min_hours
        self.task = task
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.is_full_subset = is_full_subset
        self.subsample_patients = subsample_patients

        #Hard coded means and std of age dist
        self.ages_miiv = [63.55277, 17.259564] #mean and std
        self.ages_aumc = [62.61017, 16.420369] #mean and std
        
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


        #SUBSET THE PATIENTS BY THEIR AGE CATEGORY IF needed
        if self.age_group != -1:

            if "miiv" in self.root_dir:
                age_mean = self.ages_miiv[0]
                age_std = self.ages_miiv[1]
            else:
                age_mean = self.ages_aumc[0]
                age_std = self.ages_aumc[1]

            ages_org = self.static[:,0] * age_std + age_mean

            if self.age_group == 1:
                age_min=0
                age_max=19.9
            elif self.age_group == 2:
                age_min=19.9
                age_max=45
            elif self.age_group == 3:
                age_min=45
                age_max=65
            elif self.age_group == 4:
                age_min=65
                age_max=85
            #Age group 5
            else:
                age_min=85
                age_max=120

            cond_age = np.logical_and(ages_org > age_min, ages_org <= age_max)
            print("There are " +str(cond_age.sum()) + " people in the age group " + str(self.age_group))

            self.sequence = self.sequence[cond_age]
            self.sequence_mask = self.sequence_mask[cond_age]
            self.static = self.static[cond_age]
            self.mortality = self.mortality[cond_age]



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
        elif los >= 24*8 and los<14*8:
            label = 8
        else:
            label = 9
        return label


class SensorDataset(Dataset):
    
    def __init__(self, root_dir, subject_id, split_type="train", is_cuda=True, verbose=False):
        
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.split_type = split_type
        self.is_cuda = is_cuda
        self.verbose = verbose
        
        self.load_sequence()
        
    def __len__(self):
        
        return len(self.sequence)
    
    def __getitem__(self, id_):
        
        sequence = self.sequence[id_]
        sequence_mask = np.ones(sequence.shape)
        label = self.label[id_]
        
        if self.is_cuda:
            sequence = torch.Tensor(sequence).float().cuda()
            sequence_mask = torch.Tensor(sequence_mask).long().cuda()
            label = torch.Tensor([label]).long().cuda()
        else:
            sequence = torch.Tensor(sequence).float()
            sequence_mask = torch.Tensor(sequence_mask).long()
            label = torch.Tensor([label]).long()
            
        sample = {"sequence":sequence, "sequence_mask":sequence_mask, "label":label}
        
        return sample

    def load_sequence(self):
        
        path_subject = os.path.join(self.root_dir, "subject_"+str(self.subject_id))
        
        path_sequence = os.path.join(path_subject, "timeseries_"+self.split_type+".npy")
        path_label = os.path.join(path_subject, "label_"+self.split_type+".npy")
        
        self.sequence = np.load(path_sequence, allow_pickle=True)
        self.label = np.load(path_label, allow_pickle=True)

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



