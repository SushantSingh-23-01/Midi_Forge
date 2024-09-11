import torch
import torch.nn.functional as F
from tqdm import tqdm
import pretty_midi, random

class CustomInference:
    """Inference Pipeline"""
    def __init__(self, model, tokenizer, generation_length, bpm:int|float=120, genre:str|None=None, temperature:float=1.0, top_k:int=50, verbose:bool=False) -> None:
        """
        Inference pipeline utilising top-k for generations.
        Args:
            model (_type_): pytorch model, (should be pushed to device of choice beforehand)
            tokenizer (_type_): Miditokenizers or its derivatives.
            generation_length (_type_): Recommended to be not much larger than seq_len used for triaining.
            bpm (int | float, optional): Tempo of midi to be saved. Defaults to 120.
            genre (str | None, optional): Genre of music to be generated. Defaults to None.
            temperature (float, optional): higher temperature leads to more novel and chaotic generations. Defaults to 1.0.
            top_k (int, optional): Number of top - k tokens (there logits) considered for next token prediction. Defaults to 50.
            verbose (bool): Print decoded ids, helpful for diagnoising midifile being saved. Defaults to False.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temp = temperature
        self.bpm = bpm
        self.genre = genre
        self.gen_len = generation_length
        self.top_k = top_k
        self.verbose = verbose
        self.seq_len = self.model.seq_len
        self.device =next(model.parameters()).device
        
          
    def top_k_pipline(self, token_ids:torch.Tensor, top_k:int)->torch.Tensor:
        """
        Top - k generation : Picks k - tokens with highest probs from multinomial normal distribution. Ensure it is lower than vocab_size.
        Args:
            token_ids (torch.Tensor): tokenized data transformed to tensor
            top_k (int): Number of tokens to be considered at each next token prediction.
        Returns:
            torch.Tensor: Generation.
        """
        self.model.eval()
        out = []
        with torch.inference_mode():
            for _ in tqdm(range(self.gen_len)):
                # Truncate last 'gen_len' tokens
                x_trunc = token_ids[:, -self.seq_len:]
                logits = self.model(x_trunc)
                # higher temp leads to more uniform logits making generation more novel and chaotic. (i.e less confident)
                # while lower temp makes peaky logits to become even more peakier, making generation more confident.
                probs = F.softmax(logits[:,-1,:]/self.temp, dim=-1)
                # Tokens top probs and there indices.
                topk_probs, topk_idx = torch.topk(probs, top_k)
                # sample from multionomial normal distribution
                ix = torch.multinomial(topk_probs, num_samples=1)
                next_tok = torch.gather(topk_idx,-1,ix)
                token_ids = torch.cat((x_trunc,next_tok),dim=1)
                out.append(next_tok)
            out = torch.stack(out, dim=-1)
        return out

    def generation_pipeline(self):
        """
        Important notes:\n
            - Instead of just providing genre, in practise adding a single note with step and duration helps guide model
            more into right direction. This prevents uneccessary errors which are raised due to model getting stuck at latter stages
            of generation and then repeating same token. (This leads to inproper format which raises error.)\n
        """
        start = pretty_midi.note_name_to_number('A4')
        end = pretty_midi.note_name_to_number('A6')
        idx = random.randrange(start, end)
        #in_ = [self.genre]
        in_ = [self.genre, pretty_midi.note_number_to_name(idx), 'S0.0','D0.5']
        ids = [self.tokenizer.char_to_int[i] for i in in_]
        input_ids = torch.tensor(ids, device=self.device).view(1,-1)

        out = self.top_k_pipline(input_ids, self.top_k)
        token_ids = out.view(-1,).cpu().tolist()
        decoded_tokens = self.tokenizer.detokenize(token_ids)
        if verbose is True:
            print(decoded_tokens)
        self.tokenizer.save_midifile(in_[1:]+decoded_tokens, self.bpm)
