import torch
from model import MidiModel
from mod_tokenizer import RestrictedTokenizer
from mod_trainer import RestrictedTrainer
from config import ModelArgs, TrainArgs
from mod_inference import CustomInference
    
def main():
    # Important: Format for folder : Main Dataset Folder -> Sub - Genres -> Midifiles
    midifiles = # Path to midifiles folder
    tokenizer = MidiTokenizer(midifiles)

    model = MidiModel(tokenizer.vocab_size, ModelArgs.seq_len, ModelArgs.d_model, ModelArgs.num_q_heads, ModelArgs.num_kv_heads, ModelArgs.dropout, ModelArgs.num_layers, ModelArgs.proj_fac)
    model.to(TrainArgs.device)
    
    optimizer = torch.optim.AdamW(model.parameters(),TrainArgs.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, TrainArgs.sc_gamma)
    trainer = Trainer(model, tokenizer, optimizer, scheduler, TrainArgs.batch_size, TrainArgs.epochs, TrainArgs.model_load_path, TrainArgs.model_save_path)
    # Comment out if only Inference has to be done 
    trainer.train()

    # Will throw error if model does not exsist.
    if TrainArgs.model_load_path is not None:
        trainer.load_checkpoint()
    pipline = CustomInference(model = model, tokenizer = tokenizer, gen_len = ModelArgs.seq_len, bpm = 128.0)
    pipline.generation_pipeline()
    
if __name__ == '__main__':
    main()
