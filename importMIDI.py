import joblib
import glob
import os
import pretty_midi

# PANDAS
import pandas as pd

class ImportMIDI(): 
    def __init__(self, num_files=1000): 

        all_files = glob.glob(os.path.join('..', 'lmd_aligned', '*', '*', '*', '*', '*.mid'))
        files_to_use = all_files[0:num_files]
        statistics = joblib.Parallel(n_jobs=100, verbose=50)(
            joblib.delayed(self.compute_statistics)(midi_file)
            for midi_file in files_to_use)
        # When an error occurred, None will be returned; filter those out.
        statistics = [s for s in statistics if s is not None]

        df = pd.DataFrame(statistics)

        self.imported_MIDI_data = df[df["program_numbers"].apply(self.has_all_instruments)].reset_index(drop=True)
        
 
    def get_midi_data(self): 
        return self.imported_MIDI_data 
        
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