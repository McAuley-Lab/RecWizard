from collections import deque
from functools import lru_cache
import math
import os
import random
import time
import warnings
import heapq
import numpy as np

import torch
import torch.nn as nn
import json
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_LONG = torch.long
__TORCH_AVAILABLE = True


"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    #e=nn.Embedding.from_pretrained(data, freeze=False, padding_idx=0).double()
    e = nn.Embedding(len(dictionary)+4, embedding_size, padding_idx)
    e.weight.data.copy_(torch.from_numpy(np.load('../kgsfdata/word2vec_redial.npy')))
    #nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    #e.weight=data
    #nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _create_entity_embeddings(entity_num, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(entity_num, embedding_size)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e

def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def _concept_edge_list4GCN():
    node2index=json.load(open('../kgsfdata/key2index_3rd.json',encoding='utf-8'))
    f=open('../kgsfdata/conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('../kgsfdata/stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).to(device)

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)