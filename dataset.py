import numpy as np

# PLOTTING
import collections
import math

# PYTORCH 
from torch.utils.data import Dataset

# PANDAS
import pandas as pd


class MidiDataset(Dataset):
    """MIDI dataset."""

    def __init__(self, data, data_type = "train"):
        """
        Args:None 
        """
        self.data_type = data_type
        self.all_instruments_df = data

        self.list_of_songs       = []
        self.label_list_of_songs = []
        
        self.f_list_of_songs       = []
        self.f_label_list_of_songs = []
        
        self.chunk_step_size = 1

        if self.data_type == "train" or self.data_type == "val":
            self.chunk_step_size = 10
                
        self.construct_list_of_songs()
        self.flatten_data() 
        
        
        
    def __len__(self):
        """
        Return the length of the dataset  
        """
        return len(self.f_list_of_songs)


    def __getitem__(self, idx):
        """
        Return the idx-th element of the dataset  
        """
        return self.f_list_of_songs[idx], self.f_label_list_of_songs[idx]
 
    
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
        for idx, row in self.all_instruments_df.iterrows():
            t_end = row["midi"].get_end_time()
            num_timeslices = int(t_end/T)

            # Create data array to store all of the notes in the song based on the timestep they are played in 
            data = np.zeros((num_notes * num_instruments, num_timeslices)) # rows: notes, instruments, cols: timeslices
            
            for instrument in row["midi"].instruments:
                index = self.instrument_to_index(instrument.program)
                if (index != -1):
                    for note in instrument.notes:
                        data[num_notes * index + note.pitch, math.floor(note.start/T):math.floor(note.end/T)] = 1   

            # Create chunks for a single song given its data array 
            list_of_chunks       = []
            list_of_chunk_labels = []
            start_index = 0
            end_index   = start_index + chunk_size
            while(end_index < num_timeslices):
                chunk = data[0:num_notes*(num_instruments-1),start_index:end_index]
                label = data[num_notes*(num_instruments-1):num_notes * num_instruments,start_index + int(chunk_size/2)]
                list_of_chunks.append(chunk)
                list_of_chunk_labels.append(label)
                start_index = start_index + chunk_offset
                end_index = start_index + chunk_size

            self.list_of_songs.append(list_of_chunks)
            self.label_list_of_songs.append(list_of_chunk_labels)
            
            
    def flatten_data(self): 
        """
        Flatten each set into a list of chunks. 
        """
        # Flatten data 
        for song_list in self.list_of_songs:
            for chunk in song_list:
                self.f_list_of_songs.append(chunk)
     
        for song_list in self.label_list_of_songs:
            for chunk in song_list:
                self.f_label_list_of_songs.append(chunk)

        