import symusic, pretty_midi, re, json, os, pretty_midi
import numpy as np

class RestrictedTokenizer:
    """A stripped down version on orignal transformer meant for melodies transposed in C major."""
    def __init__(self, midis_path:str, step_prec:float=0.125, oct_low:int=3, oct_high:int=7) -> None:      
        self.get_file_paths(midis_path)
        self.extract_metadata()
        self.step_prec = step_prec
        self.gen_vocab(oct_low, oct_high)
        
    def get_file_paths(self, midis_path:str):
        """
        Args:
            midis_path (str): Important: Follow the Format -> Main Folder -> Subfolders (genres) -> midis 
        """
        self.filenames = []
        for path, subdirs, files in os.walk(midis_path):
            for name in files:
                if name.endswith('.mid'):
                    self.filenames.append(os.path.join(path, name))
    
    def extract_metadata(self):
        """Gets Metadata of midifiles. (name, genre, bpm)"""
        self.metadata = []
        for f in self.filenames:
            path = os.path.normpath(f)
            name = os.path.splitext(os.path.basename(path))[0]
            genre = path.split(os.sep)[-2]
            score = symusic.Score(f, ttype=symusic.TimeUnit.second)
            bpm = score.tempos[0].qpm
            self.metadata.append({'name': name, 'genre': genre, 'bpm': round(bpm, 2)})
    
    def get_midi_data(self, midifile:str)->list[str]:
        """
        Extract data used to tokenize midifile (i.e convert to integers).
        Important terminolgies:\n
            - Step : starting timestamp of current note - starting timestamp of last note\n
            - duration : duration for which current note is being played -> (note end timestamp - note start timestamp)
        Args:
            midifile (str): Single Midifile

        Returns:
            list[str]: list[pitch, step, duration]
        """
        score = symusic.Score(midifile, ttype=symusic.TimeUnit.second)
        self.bpm = score.tempos[0].qpm
        track = score.tracks[0]
        input_ids = []
        prev_start = 0
        for note in track.notes:
            pitch = pretty_midi.note_number_to_name(note.pitch)
            step = note.start * self.bpm / 60 - prev_start
            dur = note.duration * self.bpm / 60
            # round of the note to a certain time step
            step = self.step_prec * round(step / self.step_prec)
            dur = self.step_prec * round(dur / self.step_prec)
            # cutoff note beyond certain time
            step = 8.0 if step >= 8.0 else step
            dur = 8.0 if dur >= 8.0 else dur
            input_ids.extend([f'{pitch}', f'S{step}', f'D{dur}'])
            prev_start = note.start * self.bpm / 60
        return input_ids
    
    def gen_vocab(self, oct_low, oct_high):
        """
        Generates artificial vocabulary.
        Features:\n
            - Pitch ranging from lower octave to higher octave in scale of C major.
            - Step and duration ranging from 0 (midi-step) to 16 (midi-step) or 4 bars.\n
            - Special Tokens : bos (beginning of sequence) and eos (end of sequence).\n
        """
        notes = ['A','B','C','D','E','F','G']
        pitch = []
        for o in range(oct_low, oct_high+1):
            for n in notes:
                pitch.append(f'{n}{o}')
        step = [f'S{i}' for i in np.arange(0,8.0,self.step_prec)]
        duration = [f'D{i}' for i in np.arange(0,8.0,self.step_prec)]
        
        spec_tok = ['bos','eos']
        
        self.vocab = pitch + step + duration + spec_tok
        self.char_to_int = {s:i for i, s in enumerate(self.vocab)}
        self.int_to_char = {i:s for i, s in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def tokenize(self, midifile:str)->list[int]:
        """
        Convert midifile to token_ids (integers) so it can be used as inputs to model.
        Args:
            midifile (str): path to midifile.

        Returns:
            list[int]: token_ids
        """
        input_ids = self.get_midi_data(midifile)
        token_ids = [self.char_to_int[i] for i in input_ids]
        return token_ids
    
    def detokenize(self, token_ids:list[int])->list[str]:
        """
        Converts token_ids (integers generated by a model) back to data which can be saved to midi.
        Args:
            token_ids (list[int]): 

        Returns:
            list[str]: list[pitch, step, duration]
        """
        return [self.int_to_char[i] for i in token_ids]

    def save_midifile(self, decoded_ids: list[str], tempo: float) -> None:
        """
        Saves decoded data after decoding ids generated by model to midi file. 
        Args:
            decoded_ids (list[str]): decoded token ids.
            tempo (float): beats per minute
        """
        midifile = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)
        previous_start = 0
        for i in range(0, len(decoded_ids)-3, 3):
            pitch = pretty_midi.note_name_to_number(decoded_ids[i])
            start = float(re.sub(r'[^\d\.]+',' ',decoded_ids[i+1])) * 60 / tempo + previous_start
            end = float(re.sub(r'[^\d\.]+',' ',decoded_ids[i+1])) * 60 / tempo + start
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
            previous_start = start
            instrument.notes.append(note)
        midifile.instruments.append(instrument)
        midifile.write('midiout.mid')
        
    def load_vocab_file(self):
        """Save vocabulary to json file."""
        with open(self.config.vocab_load_path,'r') as f:
            self.int_to_char = json.load(f)  
        self.int_to_char = {int(k):v for k,v in self.int_to_char.items()}
        self.char_to_int = {v:k for k, v in self.int_to_char.items()} 
        self.vocab = list(self.char_to_int.keys())
        self.vocab_size = len(self.vocab)
    
    def save_vocab_file(self):
        """Load vocabulary from json file."""
        if self.config.vocab_save_path is not None:
            with open(self.config.vocab_save_path,'w') as outfile:
                json.dump(self.int_to_char, outfile)
    
    def view_midis_info(self):
        """Used to get general data for each midifile. Helpful to diagnose any problems in midifile."""
        self.extract_metadata()
        for i in range(len(self.filenames)):
            print(f'\n{i}.Name of file :',self.metadata[i]['name'])
            print(f'bpm :', self.metadata[i]['bpm'])
            print(f'len of tokenized data: {len(self.tokenize(self.filenames[i]))}')