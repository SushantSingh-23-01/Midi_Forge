import re, json, os
from AutoTokenizer import MidiTokenizer

class BPE(MidiTokenizer):
    """Byte Pair Encoding Implementation for midi files. Inherits from our base Miditokenizer class."""
    def __init__(self, midifiles:str, final_vocab_size :int | None = None, vocab_load_path: str | None = None) -> None:
        """Main purpose of bpe is to merge frequently occuring tokens, in order to increase data to seq_len ratio.
            This allows lower seq_len model to generate more data.
        Args:
            midifiles (str): Important: Follow the Format -> Main Folder -> Subfolders (genres) -> midis
            final_vocab_size (int | None, optional): Size of vocab size after training bpe. Recommended in powers of 2. Defaults to None.
            vocab_load_path (str | None, optional): To load trained vocab. Defaults to None.
        """
        super().__init__(midifiles)
        self.filenames = []
        for path, subdirs, files in os.walk(midifiles):
            for name in files:
                if name.endswith('.mid'):
                   self.filenames.append(os.path.join(path, name))
                   
        self.orignal_vocab_size = self.vocab_size
        if final_vocab_size is not None:
            self.num_merge_counts = final_vocab_size - self.vocab_size 
        
        # dictionary {(old_mapping_tok1, old_mapping_tok_2) : merged_mapping_tok1_tok2}, neccessary for tokenization
        self.merges = {}
        if vocab_load_path is not None:
            self.load_vocab(vocab_load_path)
            self.demergify()
        
    def _extract_metadata(self, midifile: str, step_precision: float = 0.125) -> list[str]:
        return super().extract_metadata(midifile, step_precision)
    
    def _tokenize(self, midifile: str) -> list[int]:
        return super().tokenize(midifile)
    
    def tokenize_midis(self) -> list[int]:
        token_ids = []
        for filename in self.filenames:
            token_ids.extend(self._tokenize(filename))
        return token_ids
    
    def get_pair_freq(self, token_ids: list[int]) -> dict[str, int]:
        """
        Get most frequency of adjacent pair (bigrams) tokens. Order senesitive.
        Args:
            token_ids (list[int]): Tokenized data

        Returns:
            dict[str, int]: bigrams -> frequency. 
        """
        counts = {}
        for pair in zip(token_ids[:-1], token_ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        counts = dict(sorted(counts.items(), key=lambda k: k[1], reverse=True))
        return counts
    
    def merge_pairs(self, token_ids: list[int], most_freq_pair: tuple[int, int], new_index: int) -> list[int]:
        """
        Merge to integer index of tokens in most frequent bigram into a new index.\n
        Consider following Example:\n
            - (for simplicity consider text) If 't' and 'h' are frequently occuring bigram tokens, and there mapping is '1' and '2', \n
              then new mapping corresponding to 'th' is created, which is equal to : (old_vocab_size + 1).\n
            - Similarly in our case if 'C5' and 'S0.25' are most frequently occuring bigram then the are merged to 'C5 S0.25' with a new integer mapping.
            - As is visible from above case, with merging 2 tokens are now considered as 1 token.
        Args:
            token_ids (list[int]): token ids.
            most_freq_pair (tuple[int, int]): most frequently occuring bigram
            new_index (int): new index mapping to be assigned.

        Returns:
            list[int]: new token ids
        """
        new_token_ids = []
        i = 0
        while i < len(token_ids)-1:
            if i < len(token_ids) and token_ids[i] == most_freq_pair[0] and token_ids[i+1] == most_freq_pair[1]:
                new_token_ids.append(new_index)
                i += 2
            else:
                new_token_ids.append(token_ids[i])
                i += 1
        return new_token_ids
    
    def train(self, vocab_save_path: str | None = None) -> None:
        """
        Merges 'n' number of most frequently occuring bigrams, where n = final_vocab_size - orignal_vocab_size
        The merges made in process are also used for training subsequent merges.
        Args:
            vocab_save_path (str | None, optional): Save path for new integer to char/token mapping. Defaults to None.
        """
        token_ids = self.tokenize_midis()
        for i in range(self.num_merge_counts):
            counts = self.get_pair_freq(token_ids)
            freq_pair = max(counts, key= lambda x: counts[x])
            new_index = self.vocab_size + i
            token_ids = self.merge_pairs(token_ids, freq_pair, new_index)
            print(f'Merging {freq_pair} into a new token {new_index}')
            self.merges[freq_pair] = new_index
            self.vocab += [' '.join([self.int_to_char[idx] for idx in freq_pair])]
            self.int_to_char = {idx:char for idx,char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.chart_to_int = {char:idx for idx, char in enumerate(self.vocab)}
        
        if vocab_save_path is not None:
            self.save_vocab(vocab_save_path)
    
    def demergify(self):
        """Demergifies the trained vocabulary so we can get a mapping of (older_tok_1, older_tok_2) -> new_map_tok1_tok2"""
        for i in range(len(self.vocab)):
            if i > self.orignal_vocab_size:
                input_ids = re.split(' ',self.vocab[i])
                token_ids = [self.char_to_int[idx] for idx in input_ids] 
                self.merges[tuple(token_ids)] = i
            else:
                pass
            
    def tokenize(self, midifile:str) -> list[int]:
        """
        Tokenize with new merges created.
        Procedure:\n
            - Tokenize using base tokenizer, i.e. [pitch, step, duration] -> [int, int, int] mappings.\n
            - Then looping over token_ids, find bigram_counts for adjacent bigrams, take the most frequent one.\n
            - Find the mapping corresponding to the most frequent pair : old_map_tok1, old_map_tok2 -> new_map_tok1_tok2.\n
            - Merge old mappings to new mapping.
        Args:
            midifile (str): directory of midifile

        Returns:
            list[int]: token_ids / tokenized data.
        """
        # tokenize using base tokenizer
        token_ids = self._tokenize(midifile=midifile)
        while len(token_ids) >=2:
            bigram_counts = self.get_pair_freq(token_ids)
            pair = min(bigram_counts, key= lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            new_index = self.merges[pair]
            token_ids = self.merge_pairs(token_ids, pair, new_index)
            token_ids = [int(i) for i in token_ids]
        return token_ids
    
    
    def detokenize(self, token_ids: list[int]) -> list[str]:
        """
        Detokenizes data containing merged mappings.
        
        Args:
            token_ids (list[int]): tokenized_data.
        Returns:
            list[str]: decoded_tokens without merged tokens.
        """
        decoded_ids = []
        for token_id in token_ids:
            if token_id > self.orignal_vocab_size-1:
                pair = list(list(self.merges.keys())[list(self.merges.values()).index(token_id)])
                decoded_ids.extend(pair)
            else:
                decoded_ids.append(token_id)
            if len(token_ids) <= 2:
                break
        decoded_ids = super().detokenize(decoded_ids) 
        return decoded_ids
    
    def save_vocab(self, vocab_save_path: str):
        """
        Save vocab (integer to character mapping) as json.
        Args:
            vocab_save_path (str): save path
        """
        if vocab_save_path is not None:
            with open(vocab_save_path,'w') as outfile:
                json.dump(self.int_to_char, outfile)
                
    def load_vocab(self, vocab_load_path: str):
        """
        Load vocab (integer to character mapping) from json.
        Args:
            vocab_load_path (str): file load path
        """
        if vocab_load_path is not None:
            with open(vocab_load_path,'r') as f:
                self.int_to_char = json.load(f)  
            self.int_to_char = {int(k):v for k,v in self.int_to_char.items()}
            self.char_to_int = {v:k for k, v in self.int_to_char.items()} 
            self.vocab = list(self.char_to_int.keys())
            self.vocab_size = len(self.vocab)
        
    def save_midifile(self, decoded_ids: list[str], tempo: int | float) -> None:
        """Save detokenized data (pitch, step, duration) and tempo to midifile."""
        return super().save_midifile(decoded_ids, tempo)
    
    def view_midis_info(self):
        """Used to get general data for each midifile. Helpful to diagnose any problems in midifile."""
        return super().view_midis_info()

# midifiles = r'C:\Users\SUSHANT\Documents\ML_DL_Projects\Music Generation\midi_dataset'
# tokenizer = BPE(midifiles, 500)
# tokenizer.train()
# tokenizer.save_vocab('vocab.json')
