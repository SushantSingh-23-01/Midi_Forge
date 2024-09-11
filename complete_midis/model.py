import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    swiglu = swish(B(W1x+b1)) * (W2x+b2) -> (W1x+b1) * sigmoid(W1x+b1) * (W2x+b2) 
    """
    def __init__(self, dim, beta=1) -> None:
        super(SwiGLU, self).__init__()
        self.linear_gate = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))
        self.register_parameter("beta", self.beta)
    
    def forward(self, x):
        swish_gate = self.linear_gate(x) * F.sigmoid(self.beta*self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out
    
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, base=10000):
        super(RotaryPositionalEmbeddings, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
        self.dim, self.base = dim, base

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # outer product -> seq_len, dim
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.repeat_interleave(freqs, 2, -1)
            # cos, sin -> 1, 1, seq_len, dim
            self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(1)
            self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(1)
        return self.cos_cached, self.sin_cached
    
    def extra_repr(self) -> str:
        return f'in_dimension={self.dim}, base_multiplier={self.base}'
    
def rotate_half(x):
    hdim = x.shape[-1]
    x1, x2 = x[..., : hdim // 2], x[..., hdim // 2 :]
    return torch.cat((-x2, x1), dim=-1) 


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, seq_len:int, d_model:int, head_dim:int, num_q_heads:int, num_kv_heads:int, dropout:float):
        super(RoPEMultiHeadAttention, self).__init__()
        assert head_dim == d_model // num_q_heads
        causal_mask = torch.tril(torch.ones(seq_len, seq_len).view(1,1,seq_len, seq_len))
        self.register_buffer('causal_mask', causal_mask)
        
        self.query = nn.Linear(d_model , head_dim * num_q_heads)
        self.key = nn.Linear(d_model, head_dim * num_kv_heads)
        self.value = nn.Linear(d_model, head_dim * num_kv_heads)
        
        self.R = RotaryPositionalEmbeddings(head_dim)
        self.drop = nn.Dropout(dropout)
        
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_heads // num_kv_heads
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # q -> batch, seq_len, hdim * q_heads
        # k, v -> batch, seq_len, hdim * kv_heads
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # for seperating heads for indivual head-attention calculation
        # q -> batch, seq_len, num_q_heads, head_dim
        # k,v -> batch, seq_len, num_kv_heads, head_dim
        # transposing head numbers and sequence length dimension for dot product attention later
        q = q.view(batch, seq_len, self.num_q_heads, self.head_dim).transpose(1,2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1,2)
        # Grouped Query Attention, ,ight be useful.
        if self.num_q_heads != self.num_kv_heads:
            k = torch.repeat_interleave(k, repeats=self.num_q_per_kv, dim = 1)
            v = torch.repeat_interleave(v, repeats=self.num_q_per_kv, dim = 1)
        
        # applying rotary positional embedding
        cos, sin = self.R(x)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        # qr @ kr -> batch, num_kv_heads, seq_len, head_dim @ batch, num_kv_heads, head_dim, seq_len
        # attn_score -> batch, num_q_head, seq_len, seq_len
        attn_scores = qr @ kr.transpose(-2,-1) * torch.rsqrt(torch.tensor(self.head_dim))
        # Mask  future tokens
        masked_scores = torch.masked_fill(attn_scores, self.causal_mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        # normalise attention_scores
        attn_weights = F.softmax(masked_scores, dim=-1) 
        attn_weights = self.drop(attn_weights)
        # attn_out-> batch, num_q_heads, seq_len, head_dim
        attn_out = attn_weights @ v
        # batch, seq_len, d_model
        attn_out = attn_out.transpose(1,2).contiguous().view(batch, seq_len, d_model)
        return attn_out

class RoPEModelBlock(nn.Module):
    def __init__(self, seq_len:int, d_model:int, head_dim:int, num_q_heads:int, num_kv_heads:int, dropout:float, proj_fac:int=4):
        super(RoPEModelBlock, self).__init__()
        self.pre_ln = nn.LayerNorm(d_model)
        self.rmha = RoPEMultiHeadAttention(seq_len, d_model, head_dim, num_q_heads, num_kv_heads, dropout)
        self.up_proj = nn.Linear(d_model, d_model * proj_fac)
        self.swiglu = SwiGLU(d_model*proj_fac)
        self.down_proj = nn.Linear(d_model * proj_fac, d_model)
        self.post_ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        skip = x
        x = self.pre_ln(x)
        x = skip + self.rmha(x)
        skip = x
        x = self.post_ln(x)
        x = self.down_proj(self.swiglu(self.up_proj(x)))
        x += skip
        return x
    
class MidiModel(nn.Module):
    def __init__(self, vocab:int, seq_len:int, d_model:int, num_q_heads:int, num_kv_heads:int, dropout:float, num_layers:int, proj_fac:int=4):
        super(MidiModel, self).__init__()
        head_dim = d_model // num_q_heads
        self.Embedding_layer = nn.Embedding(vocab, d_model)
        self.RoPE_Attention = nn.Sequential(*[RoPEModelBlock(seq_len, d_model, head_dim, num_q_heads, num_kv_heads, dropout, proj_fac) for _ in range(num_layers)])
        self.Out_layer = nn.Linear(d_model, vocab)
        self.Out_layer.weight = self.Embedding_layer.weight

        
        self.vocab = vocab
        self.seq_len = seq_len
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.proj_fac = proj_fac
        
    def forward(self, x):
        # x -> batch, seq_len
        # embedings -> batch, seq_len, d_model
        emb = self.Embedding_layer(x)
        # attn_out -> batch, seq_len, d_model
        attn_out = self.RoPE_Attention(emb)
        # logits -> batch, seq_len, vocab_size
        logits = self.Out_layer(attn_out)
        return logits
