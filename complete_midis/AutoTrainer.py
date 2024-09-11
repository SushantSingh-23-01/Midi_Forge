import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os, json
from tqdm import tqdm

class MidiLoader:
    """Loads Midi data to a pytorch dataloader"""
    def __init__(self, tokenizer, seq_len:int, batch_size:int, device:torch.device) -> None:
        """
        Gets pytorch dataloader for a split from data extracted from midifiles.
        Args:
            tokenizer (_type_): MidiTokenizer.
            seq_len (int): seq_len at which model is to be trained. This is supposed to match the seq_len in Model definintion.
            batch_size (int): Recommended in powers of 2.
            device (torch.device): cpu or cuda.
        """
        self.tokenizer = tokenizer
        self.file_paths = self.tokenizer.filenames
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        
    def get_label_target(self, token_ids:list[int])->tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and output pairs.  
        Args:
            token_ids (list[int]): tokenized_data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: input_tensor, out_tensor
        """
        label, target = [], []
        genre_token = [self.tokenizer.char_to_int[self.tokenizer.genre]]
        for i in range(0, len(token_ids) - self.seq_len - 1):
            label.append(genre_token + token_ids[i:i+self.seq_len - 1])
            target.append(token_ids[i:i+self.seq_len])
        return torch.tensor(label, device=self.device), torch.tensor(target, device=self.device)
    
    def get_split_dataloader(self, label:torch.Tensor, target:torch.Tensor, split:str)->torch.utils.data.DataLoader:
        """
        Get Pytorch dataloader.
        Args:
            label (torch.Tensor): inputs to model.
            target (torch.Tensor): expected outputs to model.
            split (str): train, val or test.

        Returns:
            torch.utils.data.DataLoader: Dataloader for given split.
        """
        datasize = label.shape[0]
        if split == 'train':
            dataset = TensorDataset(label[:int(0.8*datasize)], target[:int(0.8*datasize)])
        elif split == 'val':
            dataset = TensorDataset(label[int(0.8*datasize):int(0.9*datasize)], target[int(0.8*datasize):int(0.9*datasize)])
        elif split == 'test':
            dataset = TensorDataset(label[int(0.9*datasize):], target[int(0.9*datasize):])
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
    
    def __call__(self, split:str)->torch.utils.data.DataLoader:   
        """
        Get Pytorch dataloader.
        Args:
            split (str): train, val or test

        Returns:
            torch.utils.data.DataLoader: dataloader for given split.
        """
        for f in self.file_paths:
            token_ids = self.tokenizer.tokenize(f)    
        x, y = self.get_label_target(token_ids)
        dataloader = self.get_split_dataloader(x, y, split)   
        return dataloader
    

class Trainer:
    """Base Trianer class."""
    def __init__(self, model, tokenizer, optimizer, scheduler, batch_size:int, epochs:int, model_save_path:str, model_load_path:str) -> None:
        """
        Trainer class for easier training.
        Args:
            model (_type_): pytorch model. (Should already be pushed to device).
            tokenizer (_type_): Midi Tokenizer & derivatives.
            optimizer (_type_): Any optimizer.
            scheduler (_type_): Any scheduler. (In case of custom ones implement step(), get_last_lr() and state_dict())
            batch_size (int): Powers of 2 recommended.
            epochs (int): Number of times entire dataset has to be seen.
            model_save_path (str): path where trained model will be saved.
            model_load_path (str): path of loading a pre-trained model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.lr = self.scheduler.get_last_lr()[0]
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path
        
        self.seq_len = model.seq_len
        self.device = next(model.parameters()).device
        
        self.loader = MidiLoader(tokenizer, self.seq_len, self.batch_size, self.device)

    def save_checkpoint(self):
        """Save model, optimizer and scheduler state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        },self.model_save_path)
    
    def load_checkpoint(self):
        """Load model, optimizer and scheduler state dict."""
        checkpoint = torch.load(self.model_load_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    def train(self):
        """Standard pytorch trainig loop."""
        if self.model_load_path is not None:
            self.load_checkpoint() 
        try:
            self.model.train()
            train_loader = self.loader('train')
            val_loader = self.loader('val')
            for epoch in range(self.epochs):
                for x, y in train_loader:
                    self.optimizer.zero_grad()
                    y_pred = self.model(x)
                    train_loss = F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
                    train_loss.backward()
                    self.optimizer.step()     
                        
                for x, y in val_loader: 
                    with torch.no_grad():
                        self.model.eval()
                        y_pred = self.model(x)
                        val_loss = F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
                self.model.train()  
                self.scheduler.step()
                if epoch % (self.epochs / 10) == 0: 
                    gpu_usage = round(torch.cuda.memory_reserved(0)/1024**3,1)
                    print('='*50)
                    print(f'\nEPochs: {epoch} | Train_loss : {train_loss.item():.6f} | validation_loss: {val_loss.item():.6f}| lr: {self.scheduler.get_last_lr()[0]:.2e}| GPU usage : {gpu_usage}\n')
                    print('='*50)
                    
        except KeyboardInterrupt:
            self.save_checkpoint()         
        self.save_checkpoint()    
                    
        except KeyboardInterrupt:
            self.save_checkpoint()         
        self.save_checkpoint()   
