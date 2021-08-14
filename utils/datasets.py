import torch, os, json, time, copy, math
import numpy as np
import PIL.Image as Image
from os.path import exists, join
from torch.utils.data import Dataset, DataLoader

def collate_fn(samples):
    return_dict = {}
    for key in samples[0]:
        return_dict[key] = torch.cat([each[key] for each in samples], dim=0)
    return return_dict


class DFGDataset(Dataset):
    def __init__(self, config, dataset_type):
        super(DFGDataset, self).__init__()
        
        self.config = config
        self.dataset_path = config['dataset_path']
        self.io = config['i/o'] # [inflow or outflow]
        self.type = dataset_type # [train, valid, test]
        self.beta = config['beta']
        self.poi_cate = config['poi_cate']
        
        self.poi = np.load(os.path.join(self.dataset_path, 'poi_{0}_{1}.npy'.format(self.poi_cate, self.beta))) # [region, cate*2]
        self.flow = np.load(os.path.join(self.dataset_path, '{0}.npy'.format(self.io))) # [region, time]
        self.checkin = np.load(os.path.join(self.dataset_path, 'checkin.npy')) # [region, time, cate]
        
        self.num_regions = self.poi.shape[0]
        np.random.seed(0)
        key_list = np.random.permutation(np.arange(self.num_regions * 48))
        if self.type == 'train':
            self.key_list = key_list[:int(len(key_list) * 0.7)]
        elif self.type == 'valid':
            self.key_list = key_list[int(len(key_list) * 0.7):int(len(key_list) * 0.85)]
        else:
            self.key_list = key_list[int(len(key_list) * 0.85):]

    def __getitem__(self, index):
        region = int(self.key_list[index] // 48)
        time = int(np.mod(self.key_list[index], 48))
        return_dict = {
            'poi' : torch.FloatTensor(self.poi[region][np.newaxis]),#[1, cate*2]
            'flow' : torch.FloatTensor(self.flow[region, time][np.newaxis, np.newaxis]),             #[1]
            'checkin' : torch.FloatTensor(self.checkin[region, time][np.newaxis]),                   #[1, cate]
            't' : torch.LongTensor(np.array(time)[np.newaxis, np.newaxis])                    #[1]
        }
        return return_dict

    def __len__(self):
        return len(self.key_list)