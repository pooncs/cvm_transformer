import math
import torch
import torch.nn as nn
from torch.nn import functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


class CVMCoreAttention(nn.Module):
    def __init__(self, d_model, n_heads, core_capacity=64):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.core_capacity = core_capacity

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, core_indices=None, core_kv=None):
        bsz, seq_len, _ = x.size()
        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        if core_kv is not None:
            k, v = core_kv
            k = k.view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
            v = v.view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        else:
            k = self.w_k(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = self.w_v(x).view(bsz, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.w_o(out)


class CVMTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, core_capacity=64, dropout=0.1):
        super().__init__()
        self.attn = CVMCoreAttention(d_model, n_heads, core_capacity)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, core_indices=None, core_kv=None):
        x = x + self.dropout(self.attn(self.norm1(x), core_indices, core_kv))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class CVMTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=8, n_layers=12, ff_dim=2048, core_capacity=64, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CVMTransformerLayer(d_model, n_heads, ff_dim, core_capacity) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, core_indices=None, core_kv=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, core_indices, core_kv)
        x = self.ln(x)
        logits = self.out(x)
        return logits


if __name__ == "__main__":
    m = CVMTransformer(vocab_size=32000, d_model=768, n_layers=2)
    inp = torch.randint(0, 32000, (2, 20))
    out = m(inp)
    print(out.shape)