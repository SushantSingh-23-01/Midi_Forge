import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os, json
from tqdm import tqdm
    
class RestrictedTrainer:
    """Restriced Trainer Class meant for melodies. Requires midi to be transposed to Scale of C Major."""
    def __init__(self, model, tokenizer, optimizer, scheduler, batch_size:int, epochs:int, model_load_path:str|None, model_save_path:str, generate_log:bool=False) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.seq_len = model.seq_len
        self.device = next(model.parameters()).device
        
        self.model_load_path = model_load_path
        self.model_save_path = model_save_path

        self.generate_log = generate_log
        
    def get_label_target(self, token_ids:list[int]):
        """
        Generates input and output pairs.
        Args:
            token_ids (list(int)): 
        Returns:
            torch.tensor : pairs on input and outputs 
        """
        x, y = [], []    
        for i in range(0, len(token_ids) - self.seq_len - 1):
            x.append(token_ids[i : i + self.seq_len])
            y.append(token_ids[i + 1 : i + self.seq_len + 1])
        return torch.tensor(x, device=self.device), torch.tensor(y, device=self.device)

    def get_split_loader(self, label:torch.Tensor, target:torch.Tensor, split:str) -> DataLoader:
        """
        Get PyTorch dataloader for train, validation or test split. Ratio of split
        Args:
            label (torch.Tensor): _description_
            target (torch.Tensor): _description_
            split (str): train, label or target
        Returns:
            DataLoader: _description_
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
    
    def get_dataloader(self, split:str) -> DataLoader:
        """
        Args:
            split (str): train, val or test
        Returns:
            DataLoader: PyTorch Dataloader
        """
        token_ids = []
        for filename in self.tokenizer.filenames:
            token_ids.extend(self.tokenizer.tokenize(filename))
        label, target = self.get_label_target(token_ids)
        dataloader = self.get_split_loader(label, target, split)
        return dataloader

    def save_checkpoint(self):
        """Saves model, optimizer and scheduler state dict."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        },self.model_save_path)
    
    def load_checkpoint(self):
        """Loads model, optimizer and scheduler state dict."""
        checkpoint = torch.load(self.model_load_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def train(self):
        """Main Function consisting of standard Pytorch training loop."""
        if self.model_load_path is not None:
            self.load_checkpoint() 
        try:
            self.model.train()
            train_loader = self.get_dataloader('train')
            val_loader = self.get_dataloader('val')
            iters = len(train_loader)
            for epoch in tqdm(range(self.epochs)):
                i = 0
                for x, y in train_loader:
                    self.optimizer.zero_grad()
                    y_pred = self.model(x)
                    train_loss = F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
                    train_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    i+=1
                with torch.no_grad():
                    self.model.eval()
                    for x, y in val_loader:
                        y_pred = self.model(x)
                        val_loss = F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
                self.model.train()  
                if epoch % (self.epochs / 10) == 0: 
                    gpu_usage = round(torch.cuda.memory_reserved(0)/1024**3,1)
                    print(f'\nEPochs: {epoch} | Train_loss : {train_loss.item():.6f} | validation_loss: {val_loss.item():.6f} | GPU usage : {gpu_usage}\n')
 
        except KeyboardInterrupt:
            self.save_checkpoint()         
        self.save_checkpoint()      
        if self.generate_log is True:
            self.save_results(train_loss, val_loss) 

    def test_model(self)->float:
        """
        Model testing.
        Returns:
            float : Average test loss for test split.
        """
        i = 0
        test_loader = self.get_dataloader('test')
        total_test_loss = 0
        for x, y in test_loader:
            i += 1
            y_pred = self.model(x)
            total_test_loss += F.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)).detach().cpu().item()
        
        avg_test_loss = total_test_loss / i
        return avg_test_loss

    def save_results(self, train_loss, val_loss):
        """Function mainly for mannual hyperparmeter tuning and book keeping"""
        result = {
            'vocab_size': self.tokenizer.vocab_size,
            'seq_len': self.model.seq_len,
            'd_model': self.model.embed_dim,
            'n_q_heads': self.model.num_q_heads,
            'n_kv_heads': self.model.num_kv_heads,
            'nlayers': self.model.num_layers,
            'dropout': self.model.dropout,
            'proj_fac': self.model.proj_factor,
            'batch_size': self.batch_size,
            'epochs':self.epochs,
            'lr': self.scheduler.get_last_lr(),
            'train_loss': round(train_loss, 2),
            'val_loss': round(val_loss, 2),
            'test_loss': round(self.test_model(), 2),
        }
        out = []
        if os.path.exists('test_results.json') is False:
            with open('test_results.json','w') as outfile:
                json.dump({'results':[]}, outfile)
        else:
            with open('test_results.json','r') as loadfile:
                out.append(json.load(loadfile))      
        out.append(result)
