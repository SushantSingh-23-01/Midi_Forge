from dataclasses import dataclass
import torch
# KEEP IN MIND : The given settings are for dataset comprising of drop melodies only (comprising ~40 to 70 notes).
# Also this are not perfect and should be modified as per need.
@dataclass
class ModelArgs: 
    seq_len:int = 96         # IMPPORTANT : Should be a multiple of 3. Depends on the number of notes in midi. 
                             # IMPORTANT : Seq len < least number of notes for a midi file in entire dataset * 3. Otherwise dataloader will throw error.
    d_model:int = 128        # embedding dimension of model, generally powers of 2 are preferred.
    num_q_heads:int = 4
    num_kv_heads:int = 4
    num_layers:int = 2       # Since model is small use either 2 or 4.
    dropout:int = 0.4        # Keep it high ~ 0.3 to 0.5, if dataset is small
    proj_fac:int = 4         # Projection factor in feedforward layer, usually 2 or 4.
    
@dataclass
class TrainArgs:
    batch_size:int = 8      # Generally preferred to keep as high as possilbe, but during testing, higher batch size lead to poor loss curves. Usually powers of 2.
    lr:float = 1e-4         # Keep it in range ~1e-4 to 1e-5 for small datasets.
    epochs:int = 10         # As high as possible without overfitting (Different from complete midi case).
    warmup_steps:int = 5    # In case of using a warmup scheduler
    sc_gamma:float = 0.9999 # In case of a scheduler with decay factor. anything < 1.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Preferred nvidia gpu / cuda.
    log_loss:bool=True      # Only meant for testing and comparing models
    model_save_path:str     # save directory of trained model.
    model_load_path:str|None = None#model_save_path    #Important: Uncomment if model exsists. Keep it None during first time of training.
