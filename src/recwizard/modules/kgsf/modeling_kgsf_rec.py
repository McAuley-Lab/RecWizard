from typing import Union, List
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import json
import pickle as pkl
from recwizard import BaseModule
from recwizard.utility.utils import WrapSingleInput
from .tokenizer_kgsf_rec import KGSFRecTokenizer
from .configuration_kgsf_rec import KGSFRecConfig
from .utils import _create_embeddings,_create_entity_embeddings, _edge_list
from .graph_utils import SelfAttentionLayer,SelfAttentionLayer_batch
from recwizard.modules.monitor import monitor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class KGSFRec(BaseModule):
    config_class = KGSFRecConfig
    tokenizer_class = KGSFRecTokenizer
    LOAD_SAVE_IGNORES = {'START', 'decoder'}

    def __init__(self, config, **kwargs):
        super().__init__(config,**kwargs)
        dictionary = config.dictionary
        self.batch_size = config.batch_size
        self.max_r_length = config.max_r_length
        self.NULL_IDX = config.padding_idx
        self.END_IDX = config.end_idx
        self.longest_label = config.longest_label

        self.pad_idx = config.padding_idx
        self.embeddings = _create_embeddings(
            dictionary, config.embedding_size, self.pad_idx, config.embedding_data
        )

        self.concept_embeddings = _create_entity_embeddings(
            config.n_concept + 1, config.dim, 0)
        self.concept_padding = 0

        self.kg = {int(k): v for k, v in config.subkg.items()}

        self.n_positions = config.n_positions

        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.self_attn = SelfAttentionLayer_batch(config.dim, config.dim)

        self.self_attn_db = SelfAttentionLayer(config.dim, config.dim)

        self.user_norm = nn.Linear(config.dim * 2, config.dim)
        self.gate_norm = nn.Linear(config.dim, 1)
        self.copy_norm = nn.Linear(config.embedding_size * 2 + config.embedding_size, config.embedding_size)
        self.representation_bias = nn.Linear(config.embedding_size, len(dictionary) + 4)

        self.info_con_norm = nn.Linear(config.dim, config.dim)
        self.info_db_norm = nn.Linear(config.dim, config.dim)
        self.info_output_db = nn.Linear(config.dim, config.n_entity)
        self.info_output_con = nn.Linear(config.dim, config.n_concept + 1)
        self.info_con_loss = nn.MSELoss(size_average=False, reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False, reduce=False)

        self.output_en = nn.Linear(config.dim, config.n_entity)

        self.embedding_size = config.embedding_size
        self.dim = config.dim

        edge_list, self.n_relation = _edge_list(self.kg, config.n_entity, hop=2)
        edge_list = list(set(edge_list))
        self.dbpedia_edge_sets = torch.LongTensor(edge_list).to(device)
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(config.n_entity, self.dim, self.n_relation, num_bases=config.num_bases)
        self.concept_edge_sets = torch.LongTensor(config.edge_set).to(device)
        self.concept_GCN = GCNConv(self.dim, self.dim)

        self.pretrain = config.pretrain
    
    def infomax_loss(self, db_nodes_features, con_user_emb, db_label, mask):
        con_emb=self.info_con_norm(con_user_emb)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.to(device).float()),dim=-1)*mask.to(device)

        return torch.mean(info_db_loss)

    def get_total_loss(self, rec_loss, info_db_loss):
        if self.pretrain:
            return info_db_loss
        else:
            return rec_loss+0.025*info_db_loss

    def forward(self, response, concept_mask, seed_sets, labels, db_vec, rec, test=True, cand_params=None, prev_enc=None, maxlen=None, bsz=None):
        print(type(concept_mask),type(seed_sets),type(db_vec),type(rec))
        if bsz == None:
            bsz = len(seed_sets)

        # graph network
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type) # entity_encoder
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets) # word_encoder

        user_representation_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if len(seed_set) == 0:
                user_representation_list.append(torch.zeros(self.dim).to(device))
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb=torch.stack(user_representation_list)
        db_con_mask=torch.stack(db_con_mask)

        graph_con_emb=con_nodes_features[concept_mask]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.to(device))
        user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb],dim=-1))
        uc_gate = F.sigmoid(self.gate_norm(user_emb))
        user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        if response is not None:
            info_db_loss = self.infomax_loss(db_nodes_features,con_user_emb,db_vec,db_con_mask)
            rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.to(device))
            rec_loss = torch.sum(rec_loss*rec.float().to(device))

            loss = self.get_total_loss(rec_loss,info_db_loss)
            labels = torch.tensor(labels)
            rec_loss = torch.tensor(rec_loss)
        else:
            loss = None
            rec_loss = None
        result = {'loss': loss, 'labels': labels, 'rec_scores': torch.tensor(entity_scores), 'rec_loss': rec_loss}
        print(result,labels)
        return result

    @WrapSingleInput
    @monitor
    def response(self, raw_input: Union[List[str], str], tokenizer, return_dict=False, topk=3):
        movieIds_lst = []
        movieNames_lst = []
        for raw_single in raw_input:
            inputs = tokenizer.encode(raw_single)
            logits = self.forward(**inputs)['rec_scores']
            movieIds, movieNames = tokenizer.decode(logits)
            movieIds_lst.append(movieIds)
            movieNames_lst.append(movieNames)
        if return_dict:
            return {
                'logits': logits,
                'movieIds': movieIds_lst,
                'movieNames': movieNames_lst,
            }
        return movieNames_lst
        