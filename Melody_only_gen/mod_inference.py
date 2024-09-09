import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

class CustomInference:
    def __init__(self, model, tokenizer, gen_len:int, bpm:int|float=120.0, temperature:float=1.0, top_k:int=50, midifile_save_path:str|None=None, verbose:bool=False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gen_len = gen_len
        self.temp = temperature
        self.bpm = bpm
        self.top_k = top_k
        self.seq_len = model.seq_len
        self.device =next(model.parameters()).device
        self.midifile_save_path = midifile_save_path if midifile_save_path is not None else ''
        self.verbose = verbose
          
    def top_k_pipline(self, token_ids, top_k):
        """Top-k Sampling"""
        self.model.eval()
        out = []
        with torch.inference_mode():
            for _ in tqdm(range(self.gen_len)):
                x_trunc = token_ids[:, -self.seq_len:]
                logits = self.model(x_trunc)
                probs = F.softmax(logits[:,-1,:]/self.temp, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, top_k)
                ix = torch.multinomial(topk_probs, num_samples=1)
                next_tok = torch.gather(topk_idx,-1,ix)
                token_ids = torch.cat((x_trunc,next_tok),dim=1)
                out.append(next_tok)
            out = torch.stack(out, dim=-1)
        return out

    def generation_pipeline(self):
        notes = ['A','B','C','D','E','F','G']
        pitch = []
        # Restricting to 4 to 7th octave as most melodies lie within those range. 
        for o in range(4, 8):
            for n in notes:
                pitch.append(f'{n}{o}')
        in_ = [random.choice(pitch), 'S0.0','D0.25']
        ids = [self.tokenizer.char_to_int[i] for i in in_]
        input_ids = torch.tensor(ids, device=self.device).view(1,-1)

        out = self.top_k_pipline(input_ids, self.top_k)
        token_ids = out.view(-1,).cpu().tolist()
        decoded_tokens = self.tokenizer.detokenize(token_ids)
        if self.verbose is True:
          print(decoded_tokens)
        self.tokenizer.save_midifile(decoded_tokens, self.bpm)
