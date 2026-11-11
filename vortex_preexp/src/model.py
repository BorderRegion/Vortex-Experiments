from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from . import vortex, spectral
from .quant import quantize


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 512
    heads: int = 4
    layers: int = 2
    context: int = 512
    method: str = 'rope_safe'
    backend: str = 'sdpa'


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.cache = {}
        self.quant = None
        self.capture = None

    def position(self, x, offset=0):
        method = self.cfg.method
        key = (method, x.shape[1], offset, str(x.device))
        if key not in self.cache:
            pos = torch.arange(offset, offset+x.shape[1])
            self.cache[key] = (vortex.indices(pos, method != 'ring_plain', x.device)
                               if method in ('vortex', 'ring_plain') else
                               spectral.rotation_tables(pos, method, x.device))
        table = self.cache[key]
        return vortex.apply(x, *table) if method in ('vortex', 'ring_plain') else spectral.rotate(x, table)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).reshape(b, s, 3, self.cfg.heads, 128).unbind(2)
        if self.capture is not None:
            self.capture.append((q.detach().float().cpu(), k.detach().float().cpu()))
        if self.quant and self.quant.get('order') == 'pre':
            if self.quant['scope'] == 'qk': q = quantize(q, self.quant['group'], self.quant.get('kind','int4'))
            k = quantize(k, self.quant['group'], self.quant.get('kind','int4'))
        q, k = self.position(q), self.position(k)
        if self.quant and self.quant.get('order','post') == 'post':
            if self.quant['scope'] == 'qk': q = quantize(q, self.quant['group'], self.quant.get('kind','int4'))
            k = quantize(k, self.quant['group'], self.quant.get('kind','int4'))
        q, k, v = [z.transpose(1, 2).contiguous() for z in (q,k,v)]
        if self.cfg.backend == 'sdpa':
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            score = q.float()@k.float().transpose(-1,-2)/128**0.5
            mask = torch.ones(s, s, device=x.device, dtype=torch.bool).triu(1)
            score = score.masked_fill(mask, float('-inf'))
            y = score.softmax(-1).to(v.dtype)@v
        return self.proj(y.transpose(1,2).contiguous().reshape(b,s,d))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model,4*cfg.d_model,bias=False), nn.GELU(),
                                 nn.Linear(4*cfg.d_model,cfg.d_model,bias=False))
    def forward(self,x):
        x = x+self.attn(self.ln1(x))
        return x+self.ffn(self.ln2(x))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model == cfg.heads*128
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.output.weight = self.embedding.weight
        self.apply(self._init)
    def _init(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
    def set_quant(self,quant):
        for block in self.blocks: block.attn.quant = quant
    def forward(self,x,labels=None):
        x = self.embedding(x)
        for block in self.blocks: x = block(x)
        logits = self.output(self.norm(x))
        loss = F.cross_entropy(logits.float().reshape(-1,self.cfg.vocab_size),labels.reshape(-1),ignore_index=-100) if labels is not None else None
        return logits,loss
