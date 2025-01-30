#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Derived from OpenAI Whisper model file:
# https://github.com/openai/whisper/blob/main/whisper/model.py

from typing import Optional
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from flash_attn import flash_attn_func

from espnet2.speechlm.module.abs_transformer import AbsTransformer
from espnet2.speechlm.net_utils import install_kv_cache_hook

from .rotary import apply_rotary_emb
# try:
#     from apex.normalization import FusedRMSNorm as RMSNorm 
# except ModuleNotFoundError:
#     print("No fused RMSNorm")
    # from .rms_norm import RMSNorm
from .rms_norm import RMSNorm

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # return super().forward(x) # For full BF16 training
        return super().forward(x.float()).type(x.dtype)  # For AMP / FP32 training


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return freqs

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        causal: bool = False,
        qk_norm: bool = False,
        dropout: float = 0.0,
        depth=None,
    ):
        super().__init__()
        assert n_state % n_head == 0
        assert n_state % n_head % 2 == 0
        # raise SystemError(n_head, n_state, causal, qk_norm, dropout, depth)
        # (4, 512, True, True, 0.0, 0)
        self.n_head = n_head
        self.query = Linear(n_state, n_state, bias=False)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state, bias=False)
        self.out = Linear(n_state, n_state, bias=False)
        self.causal = causal
        self.dropout = dropout
        self.n_state = n_state
        self.head_dim = self.n_state // self.n_head // 2
        self.scaling = self.head_dim ** -0.5

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("Install torch 2.0.1+ to support Flash Attention")

        self.flash_attn_func = flash_attn_func


    def build_rel_pos(self, x, start_pos=0, rope_theta: float = 10000.0, max_seq_len: int = -1):
        # not sure if it is correct
        max_seq_len = 2048
        if max_seq_len < x.size(1):
            raise ValueError(max_seq_len, x.size(1))
        _precomputed_freqs_cis = precompute_freqs_cis(
            self.head_dim, max_seq_len, rope_theta
        ).to(x.device)

        cos = torch.cos(_precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(_precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # x = x.to(device='cuda:0')
        # raise SystemError(x.device)
        q = self.query(x)
        # raise SystemError(kv_cache) # None
        # training is not ; inference why it is using? it is not cross attention
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            if xa is not None:
                raise SystemError("xa should be None which mean it is cross attention")
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]
            raise SystemError("In our case, xa should be None which mean it is cross attention")

        # k = self.key(x)
        # v = self.value(x)
        # b x t x d

        # prepare for rotary positional embedding
        rel_pos = self.build_rel_pos(v, start_pos=0)

        wv = self.qkv_attention(q, k, v, mask, rel_pos)

        return self.out(wv)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, rel_pos=None
    ):
        if self.causal and mask is not None:
            raise ValueError("mask is not allowed when the attention is causal")

        if self.causal and q.size(1) == k.size(1):
            causal = True
        else:
            causal = False
            # raise SystemError(causal, q.size(), k.size(), "q k seq leng is not same, it should not happen in our case")
        # batch size, sequence length, hidden size
        # raise SystemError(q.shape, k.shape, v.shape, causal) # torch.Size([1, 626, 512]) torch.Size([1, 626, 512]) torch.Size([1, 626, 512]) False
        q = q.view(*q.shape[:2], 2*self.n_head, self.head_dim) # b x t x h x d
        k = k.view(*k.shape[:2], 2*self.n_head, self.head_dim) 
        v = v.view(*v.shape[:2], self.n_head, 2, self.head_dim) # b x t x h x d

        # apply Rotary Positional Embedding
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        q = q.reshape(*q.shape[:2], self.n_head, 2, self.head_dim)
        k = k.reshape(*k.shape[:2], self.n_head, 2, self.head_dim)


        q1, q2 = q[:, :, :, 0], q[:, :, :, 1] # b x t x h x d
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # flash_attn_func input is b x t x h x d
        # return (batch_size, seqlen, nheads, headdim)
        # scaled_dot_product_attention input is b x h x t x d
        # return (batch_size, nheads, seqlen, headdim)

        attn11 = flash_attn_func(q1, k1, v1, causal=True)
        attn12 = flash_attn_func(q1, k1, v2, causal=True)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = flash_attn_func(q2, k2, v1, causal=True)
        attn22 = flash_attn_func(q2, k2, v2, causal=True)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        wv = attn1 - lambda_full * attn2
        
        wv = self.subln(wv) # layer norm (norm feats)
        wv = wv * (1 - self.lambda_init)
        # print(wv.shape, v.shape, q.shape, k.shape)
        wv = wv.reshape(*wv.shape[:2], self.n_head * 2 * self.head_dim)
        # raise SystemError(wv.shape) # torch.Size([11, 860, 512]) batch_size, seqlen, hidden_size
        return wv


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        causal: bool = False,
        qk_norm: bool = False,
        dropout: float = 0.0,
        depth=None,
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            n_state,
            n_head,
            causal=causal,
            qk_norm=qk_norm,
            dropout=dropout,
            depth=depth,
        )
        self.attn_ln = LayerNorm(n_state)
        self.attn_dropout = nn.Dropout(p=dropout)

        self.cross_attn = (
            MultiHeadAttention(
                n_state,
                n_head,
                causal=False,
                qk_norm=qk_norm,
                dropout=dropout,
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.cross_attn_dropout = nn.Dropout(p=dropout) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.mlp_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn_dropout(
            self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        )
        if self.cross_attn:
            x = x + self.cross_attn_dropout(
                self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
            )
        x = x + self.mlp_dropout(self.mlp(self.mlp_ln(x)))
        return x


class DiffROPETransformerDecoder(AbsTransformer):
    def __init__(
        self,
        token_bias: dict,
        n_ctx: int = 128,
        n_state: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        causal: bool = True,
        qk_norm: bool = False,
        dropout: float = 0.0,
        layer_class=ResidualAttentionBlock,
    ):
        super().__init__()
        # self.pos_emb = nn.Embedding(n_ctx, n_state)
        # cross attention is always false
        self.blocks = nn.ModuleList(
            [
                layer_class(
                    n_state=n_state,
                    n_head=n_head,
                    cross_attention=False,
                    causal=causal,
                    qk_norm=qk_norm,
                    dropout=dropout,
                    depth=d,
                )
                for d in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        self.causal = causal
        self.d_model = n_state
        self._n_ctx = n_ctx
        
        self.kv_cache = None
        self.hooks = None

    def forward(self, x: Tensor, mask: torch.Tensor = None):
        if self.causal and mask is not None:
            raise ValueError("Causal Transformer dones't allow mask")
        
        # offset = next(iter(self.kv_cache.values())).shape[1] if self.kv_cache else 0
        # x = x + self.pos_emb.weight[offset : offset + x.shape[1]].unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask=mask, kv_cache=self.kv_cache)

        x = self.ln(x)
        return x
    
    def init(self):
        self.kv_cache, self.hooks = install_kv_cache_hook(
            self.blocks, 
            self.kv_cache,
            attn_module=MultiHeadAttention,
        )

    def reset(self):
        for h in self.hooks:
            h.remove()
        self.kv_cache = None
        self.hooks = None
    
    def select_state(self, index):
        if self.kv_cache is None:
            raise ValueError("Transformer is not initialized or doesn't have kv_cache")
        
        for k, v in self.kv_cache.items():
            self.kv_cache[k] = v[index]
    
    @property
    def n_ctx(self):
        return self._n_ctx
        
