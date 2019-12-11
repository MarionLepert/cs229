import numpy as np

# PLOTTING
import collections
import math

# PYTORCH 
from torch.utils.data import Dataset

# PANDAS
import pandas as pd

import h5py
import pickle as pkl


class MidiSavedDataset(Dataset):
    """MIDI dataset."""

    def __init__(self, data_type = "train"):
        """
        Args:None 
        """
        self.data_type = data_type
            
        self.dict_of_where_to_look = {}
        
        self.filename = data_type + '.hdf5'
        self.filename_labels = data_type + "Labels.hdf5"
            
        self.hf_read = None
        self.hf_read_labels = None
#         self.hf_read        = h5py.File(filename, 'r')
#         self.hf_read_labels = h5py.File(filename_labels, 'r')
        if (data_type == 'train'): 
            with open("trainOther.pkl", "rb") as pf:
                self.length, self.dict_of_where_to_look = pkl.load(pf)
        elif (data_type == 'val'):
            with open("valOther.pkl", "rb") as pf:
                self.length, self.dict_of_where_to_look = pkl.load(pf)
        
    def __del__(self):
        self.hf_read.close()
        self.hf_read_labels.close()
        
    def __len__(self):
        """
        Return the length of the dataset  
        """
#         if (self.data_type == "train"): 
#             return 1507807
#         else: 
#             return 184477
        if (self.data_type == "train"): 
            return 30000
        else: 
            return 3750
#         return self.length


    def __getitem__(self, idx):
        """
        Return the idx-th element of the dataset  
        """
        data = []
        labels = []
        
        if self.hf_read is None:
            self.hf_read = h5py.File(self.filename, 'r')
        if self.hf_read_labels is None:
            self.hf_read_labels = h5py.File(self.filename_labels, 'r')
               
        # Data
        song, chunk = self.dict_of_where_to_look[idx]
#         print("Song: ", song, ", Chunk: ", chunk)
        data = self.hf_read[str(song)][:, chunk[0]:chunk[1]]
        # Labels
        song, chunk = self.dict_of_where_to_look[idx]
        mid_index = chunk[0] + (chunk[1]-chunk[0])/2.0
        labels = self.hf_read_labels[str(song)][:,mid_index]
        
        
        return data, labels
    

    
