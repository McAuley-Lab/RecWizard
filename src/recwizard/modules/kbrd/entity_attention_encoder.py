""" This file is adapted from the KBRD original implementation: https://github.com/THUDM/KBRD
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=-1)
        return torch.matmul(attention, h)


class KBRD(nn.Module):
    def __init__(
        self, n_entity, n_relation, sub_n_relation, dim, edge_idx, edge_type, num_bases
    ):
        super(KBRD, self).__init__()

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.sub_n_relation = sub_n_relation
        self.dim = dim

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        nn.init.kaiming_uniform_(self.entity_emb.weight.data)

        self.self_attn = SelfAttentionLayer(self.dim, self.dim)
        self.output = nn.Linear(self.dim, self.n_entity)

        self.rgcn = RGCNConv(
            self.n_entity, self.dim, self.sub_n_relation, num_bases=num_bases
        )

        self.edge_idx = nn.Parameter(edge_idx, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor):
        # [batch size, dim]
        u_emb, nodes_features = self.user_representation(input_ids, attention_mask)
        return F.linear(u_emb, nodes_features, self.output.bias)

    def user_representation(self, input_ids, attention_mask):
        nodes_features = self.rgcn(None, self.edge_idx, self.edge_type)

        user_representation_list = []
        for input_ids_, attention_mask_ in zip(input_ids, attention_mask):
            valid_num = attention_mask_.sum() if len(attention_mask_) else 0
            if valid_num == 0:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                continue
            user_representation = nodes_features[input_ids_[:valid_num]]
            user_representation = self.self_attn(user_representation)
            user_representation_list.append(user_representation)
        return torch.stack(user_representation_list), nodes_features
