# MIDI FILES
import pretty_midi
import numpy as np
import joblib
import glob

# PLOTTING
import collections
import os
import math

# PYTORCH 
from torch.utils.data import Dataset

# PANDAS
import pandas as pd

class MidiDataset(Dataset):
    """MIDI dataset."""

    def __init__(self, num_files=5000,data_type="train"):
        """
        Args:None 
        """
        self.num_files  = num_files
        self.data_type  = data_type
        self.train_size = 0.60
        self.val_size   = 0.30
        self.test_size  = 0.10

        self.list_of_songs         = []
        self.f_train_list_of_songs = []
        self.f_val_list_of_songs   = []
        self.f_test_list_of_songs  = []
        
        self.label_list_of_songs         = []
        self.label_f_train_list_of_songs = []
        self.label_f_val_list_of_songs   = []
        self.label_f_test_list_of_songs  = []

        self.import_midi_files()
        self.construct_list_of_songs()
        self.split_and_flatten_data() 
        
    def __len__(self):
        """
        Return the length of the appropriate dataset  
        """
        if (self.data_type == "train"):
            return len(self.f_train_list_of_songs)
        elif (self.data_type == "val"):
            return len(self.f_val_list_of_songs)
        elif (self.data_type == "test"): 
            return len(self.f_test_list_of_songs)

    def __getitem__(self, idx):
        """
        Return the idx-th element of the dataset  
        """
        if (self.data_type == "train"):
            return self.f_train_list_of_songs[idx], self.label_f_train_list_of_songs[idx]
        elif (self.data_type == "val"):
            return self.f_val_list_of_songs[idx], self.label_f_val_list_of_songs[idx]
        elif (self.data_type == "test"): 
            return self.f_test_list_of_songs[idx], self.label_f_test_list_of_songs[idx]
        
    
    def compute_statistics(self, midi_file):
        """
        Given a path to a MIDI file, compute a dictionary of statistics about it

        Parameters
        ----------
        midi_file : str
            Path to a MIDI file.

        Returns
        -------
        statistics : dict
            Dictionary reporting the values for different events in the file.
        """
        # Some MIDI files will raise Exceptions on loading, if they are invalid.
        # We just skip those.
        try:
            pm = pretty_midi.PrettyMIDI(midi_file)
            # Extract informative events from the MIDI file
            return {'n_instruments': len(pm.instruments),
                    'program_numbers': [i.program for i in pm.instruments if not i.is_drum],
                    'key_numbers': [k.key_number for k in pm.key_signature_changes],
                    'tempos': list(pm.get_tempo_changes()[1]),
                    'time_signature_changes': pm.time_signature_changes,
                    'end_time': pm.get_end_time(),
                    'lyrics': [l.text for l in pm.lyrics],
                    'path': midi_file,
                    'midi': pm}
        # Silently ignore exceptions for a clean presentation (sorry Python!)
        except Exception as e:
            pass
        
    def has_all_instruments(self, program_numbers):
        """
        Checks if the program numbers contain all four desired instruments
        
        Parameters
        ----------
        program numbers : list of program numbers

        Returns
        -------
        True  : if program numbers contains all four desired instruments
        False : otherwise
        """
        piano_program_numbers = set([0, 1, 2, 3, 4])
        guitar_program_numbers = set([25, 26, 27, 28, 29])
        bass_program_numbers = set([33, 34, 35, 36, 37, 38, 52])
        string_program_numbers = set([41, 42, 43, 49, 50, 51])

        return not set(program_numbers).isdisjoint(piano_program_numbers) and \
               not set(program_numbers).isdisjoint(guitar_program_numbers) and \
               not set(program_numbers).isdisjoint(bass_program_numbers) and \
               not set(program_numbers).isdisjoint(string_program_numbers) 
    
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
        
    def import_midi_files(self): 
        """
        Import the midi files and output an array of all the midi files with 

        """
        all_files = glob.glob(os.path.join('..', 'lmd_aligned', '*', '*', '*', '*', '*.mid'))
        
        files_to_use = all_files[0:self.num_files]
        
        # Compute statistics about every file in our collection in parallel using joblib
        # We do things in parallel because there are tons so it would otherwise take too long!
        statistics = joblib.Parallel(n_jobs=100, verbose=50)(
            joblib.delayed(self.compute_statistics)(midi_file)
            for midi_file in files_to_use)
        # When an error occurred, None will be returned; filter those out.
        statistics = [s for s in statistics if s is not None]
        
        df = pd.DataFrame(statistics)
        self.all_instruments_df = df[df["program_numbers"].apply(self.has_all_instruments)].reset_index(drop=True)
        

    def construct_list_of_songs(self):
        """
        Import an array of midi files and output a list of songs 
        where each song is composed of a list of chunks and each 
        chunk is a numpy array. 

        """
        T = 0.010   # Timestep (s)
        chunk_size      = 50
        chunk_offset    = 10
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
            
            
    def split_and_flatten_data(self): 
        """
        Split a list of songs into train, val, and test sets. 
        Flatten each set into a list of chunks. 
        """
        # Split data 
        num_songs  = len(self.list_of_songs)
        val_index  = math.floor(num_songs*self.train_size)
        test_index = math.floor(num_songs*(1-self.test_size))

        # X 
        train_list_of_songs = self.list_of_songs[0:val_index]
        val_list_of_songs   = self.list_of_songs[val_index: test_index]
        test_list_of_songs  = self.list_of_songs[test_index:]
        
        # Y 
        label_train_list_of_songs = self.label_list_of_songs[0:val_index]
        label_val_list_of_songs   = self.label_list_of_songs[val_index: test_index]
        label_test_list_of_songs  = self.label_list_of_songs[test_index:]

        # Flatten data 
        # X
        for song_list in train_list_of_songs:
            for chunk in song_list:
                self.f_train_list_of_songs.append(chunk)

        for song_list in val_list_of_songs:
            for chunk in song_list:
                self.f_val_list_of_songs.append(chunk)

        for song_list in test_list_of_songs:
            for chunk in song_list:
                self.f_test_list_of_songs.append(chunk)
                
        # Y       
        for song_list in label_train_list_of_songs:
            for chunk in song_list:
                self.label_f_train_list_of_songs.append(chunk)

        for song_list in label_val_list_of_songs:
            for chunk in song_list:
                self.label_f_val_list_of_songs.append(chunk)

        for song_list in label_test_list_of_songs:
            for chunk in song_list:
                self.label_f_test_list_of_songs.append(chunk)

