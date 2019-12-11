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

class MidiToFile(Dataset):
    """MIDI dataset."""

    def __init__(self, data, data_type = "train"):
        """
        Args:None 
        """
        self.data_type = data_type
        self.dataset_name = data_type
        self.all_songs_df = data

        self.list_of_songs       = []
        self.label_list_of_songs = []
        
        self.chunk_step_size = 1
        self.chunk_size = 50

        if self.data_type == "train" or self.data_type == "val":
            self.chunk_step_size = 10
            
        self.length = 0
            
        self.dict_of_where_to_look = {}
        
        filename = data_type + '.hdf5'
        filename_labels = data_type + "Labels.hdf5"
            
        self.construct_list_of_songs()
        
        print("List of songs length: ", len(self.list_of_songs))
        print("List of labels length: ", len(self.label_list_of_songs))
        
        with h5py.File(filename, 'w') as hf:
            self.save_data(hf, self.list_of_songs)
            
        with h5py.File(filename_labels, 'w') as hf: 
            self.save_data(hf, self.label_list_of_songs)
            
        picklename = data_type + "Other.pkl"
        with open(picklename, "wb") as pf:
            pkl.dump((self.length, self.dict_of_where_to_look), pf)
           
 
    def instrument_to_index(self, instrument):
        """
        Return the index of the category the instrument belongs to 
        
        Parameters
        ----------
        Instrument : instrument program number

        Returns
        -------
        0  : instrument program number is in the piano category 
        1  : instrument program number is in the guitar category
        2  : instrument program number is in the string category
        3  : instrument program number is in the bass category
        -1 : instrument program number is not in our of the four desired categories
        """
        piano_program_numbers  = set([0, 1, 2, 3, 4])
        guitar_program_numbers = set([25, 26, 27, 28, 29])
        bass_program_numbers   = set([33, 34, 35, 36, 37, 38, 52])
        string_program_numbers = set([41, 42, 43, 49, 50, 51])

        if instrument in piano_program_numbers:
            return 0
        elif instrument in guitar_program_numbers:
            return 1
        elif instrument in string_program_numbers:
            return 2 
        elif instrument in bass_program_numbers:
            return 3
        else:
            return -1
        
    def construct_list_of_songs(self):
        """
        Import an array of midi files and output a list of songs 
        where each song is composed of a list of chunks and each 
        chunk is a numpy array. 

        """
        T = 0.010   # Timestep (s)
        chunk_size      = 50
        chunk_offset    = self.chunk_step_size
        num_notes       = 128
        num_instruments = 4
        
        # Iterate through every midi and extract chunks in each song 
        for idx, row in self.all_songs_df.items():
            t_end = row.get_end_time()
            num_timeslices = int(t_end/T)

            # Create data array to store all of the notes in the song based on the timestep they are played in 
            data = np.zeros((num_notes * num_instruments, num_timeslices)) 
            
            for instrument in row.instruments:
                index = self.instrument_to_index(instrument.program)
                if (index != -1):
                    for note in instrument.notes:
                        data[num_notes * index + note.pitch, math.floor(note.start/T):math.floor(note.end/T)] = 1   
            
            self.list_of_songs.append(data[0:num_notes*(num_instruments-1), :])
            self.label_list_of_songs.append(data[num_notes*(num_instruments-1):num_notes*num_instruments, :])
        
            
    def save_data(self,hf, list_of_items): 
        print("Length of list: ", len(list_of_items))
        chunk_idx = 0
        for idx, song in enumerate(list_of_items):
            for chunk in range(0, song.shape[1]-self.chunk_size, 10):
                self.dict_of_where_to_look[chunk_idx] = (idx, (chunk, chunk+self.chunk_size))
                chunk_idx += 1
            self.write_song_to_h5(str(idx), song, hf)
            print("Song index: ", idx)

        self.length = chunk_idx
        print("Num chunks: ", self.length)
            
    def write_song_to_h5(self, idx, song, hf): 
        # idx corresponds to the song index written as a string
#         import pdb; pdb.set_trace()
        hf.create_dataset(idx, data=song)
    
    def save_dict(self, hf):
        hf.create_dataset('Dict', data=self.dict_of_where_to_look)
        
    def save_length(self, hf): 
        hf.create_dataset('Length', data=self.length)
                       

    
    