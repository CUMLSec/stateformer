import torch
import math
from torch import nn


def init_(t, dim=None):
    dim = dim if dim is not None else t.shape[-1]
    std = 1. / math.sqrt(dim)
    return nn.init.normal_(t, mean=0, std=std)


def expand_dim(t, dim, k, unsqueeze=False):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def list_subtract(l, r):
    return [el for el in l if el not in set(r)]


def fetch_pkm_value_parameters(module):
    params = []
    for m in module.modules():
        if isinstance(m, PKM):
            params.append(m.values.weight)
    rest = list_subtract(module.parameters(), params)
    return params, rest


def fetch_optimizer_parameters(module, pkm_learning_rate=1e-2):
    pkm_params, rest = fetch_pkm_value_parameters(module)
    return [{'params': rest}, {'params': pkm_params, 'lr': pkm_learning_rate}]


class MaskedBatchNorm1D(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, mask=None):
        b, t, d = x.shape
        has_mask = mask is not None

        if has_mask:
            initial_x = x
            mask = mask.unsqueeze(-1)
            x = x.masked_select(mask)

        shape = x.shape
        x = x.reshape(-1, d)
        x = self.fn(x)
        x = x.reshape(*shape)

        if has_mask:
            x = initial_x.masked_scatter(mask, x)

        return x


class PKM(nn.Module):
    def __init__(self, dim, heads=4, num_keys=128, topk=32, dim_head=256, input_dropout=0., query_dropout=0.,
                 value_dropout=0.):
        super().__init__()
        assert (dim % heads == 0), 'dimension must be divisible by number of heads'
        self.topk = topk
        self.heads = heads
        self.num_keys = num_keys

        dim_query = dim_head * heads
        self.to_queries = nn.Linear(dim, dim_query, bias=False)
        self.norm = MaskedBatchNorm1D(nn.BatchNorm1d(dim_query))

        self.keys = nn.Parameter(torch.zeros(heads, num_keys, 2, dim_head // 2))
        self.values = nn.EmbeddingBag(num_keys ** 2, dim, mode='sum')
        init_(self.keys)
        init_(self.values.weight)

        self.input_dropout = nn.Dropout(input_dropout)
        self.query_dropout = nn.Dropout(query_dropout)
        self.value_dropout = nn.Dropout(value_dropout)

    def forward(self, x, input_mask=None, **kwargs):
        b, t, e, h = *x.shape, self.heads
        x = self.input_dropout(x)

        queries = self.to_queries(x)
        queries = self.norm(queries, mask=input_mask)
        queries = self.query_dropout(queries)

        queries = queries.chunk(2, dim=-1)
        queries = torch.stack(queries).reshape(2, b, t, h, -1)

        dots = torch.einsum('pbthd,hnpd->bthpn', queries, self.keys)
        scores, indices = dots.topk(k=self.topk, dim=-1)
        scores, indices = map(lambda x: x.chunk(2, dim=3), (scores, indices))

        all_topk = self.topk ** 2
        shape = (b, t, h, all_topk)

        all_scores = (
                scores[0][..., :, None] +
                scores[1][..., None, :]
        ).reshape(*shape)

        all_indices = (
                indices[0][..., :, None] * self.num_keys +
                indices[1][..., None, :]
        ).reshape(*shape)

        final_topk, final_indices = all_scores.topk(self.topk, dim=-1)
        value_indices = all_indices.gather(-1, final_indices)

        attn = final_topk.softmax(dim=-1)

        value_indices, attn = map(lambda x: x.reshape(-1, self.topk * h), (value_indices, attn))

        out = self.values(value_indices, per_sample_weights=attn)
        out = self.value_dropout(out)
        return out.reshape(b, t, e)
