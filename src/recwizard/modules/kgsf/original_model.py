""" This file is adapted from KGSF original implementation: https://github.com/Lancelot39/KGSF/blob/master/model.py
"""

from typing import Union, List

import torch

from torch import nn
from torch.nn import functional as F

import numpy as np

from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv

from recwizard.modules.kgsf.original_utils import (
    _create_embeddings,
    _create_entity_embeddings,
    _edge_list,
)
from recwizard.modules.kgsf.original_gcn import SelfAttentionLayer, SelfAttentionLayer_batch
from recwizard.modules.kgsf.original_transformer import TransformerEncoder, TransformerDecoderKG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecModel(nn.Module):

    def __init__(self, config, pretrained_word_embedding, dbpedia_edge_sets, concept_edge_sets, **kwargs):
        super().__init__()
        self.vocab_size = pretrained_word_embedding.size(0)

        self.batch_size = config.batch_size
        self.max_r_length = config.max_r_length
        self.NULL_IDX = config.padding_idx
        self.END_IDX = config.end_idx
        self.longest_label = config.longest_label

        self.pad_idx = config.padding_idx
        self.embeddings = _create_embeddings(
            self.vocab_size, config.embedding_size, self.pad_idx, pretrained_word_embedding
        )

        self.concept_embeddings = _create_entity_embeddings(config.n_concept + 1, config.dim, 0)
        self.concept_padding = 0

        self.n_positions = config.n_positions

        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.self_attn = SelfAttentionLayer_batch(config.dim, config.dim)

        self.self_attn_db = SelfAttentionLayer(config.dim, config.dim)

        self.user_norm = nn.Linear(config.dim * 2, config.dim)
        self.gate_norm = nn.Linear(config.dim, 1)
        self.copy_norm = nn.Linear(config.embedding_size * 2 + config.embedding_size, config.embedding_size)
        self.representation_bias = nn.Linear(config.embedding_size, self.vocab_size)

        self.info_con_norm = nn.Linear(config.dim, config.dim)
        self.info_db_norm = nn.Linear(config.dim, config.dim)
        self.info_output_db = nn.Linear(config.dim, config.n_entity)
        self.info_output_con = nn.Linear(config.dim, config.n_concept + 1)
        self.info_con_loss = nn.MSELoss(size_average=False, reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False, reduce=False)

        self.output_en = nn.Linear(config.dim, config.n_entity)

        self.embedding_size = config.embedding_size
        self.dim = config.dim

        self.dbpedia_edge_sets = dbpedia_edge_sets

        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(config.n_entity, self.dim, config.n_relation, num_bases=config.num_bases)
        self.concept_edge_sets = concept_edge_sets
        self.concept_GCN = GCNConv(self.dim, self.dim)

        self.pretrain = config.pretrain

    def infomax_loss(self, db_nodes_features, con_user_emb, db_label, mask):
        con_emb = self.info_con_norm(con_user_emb)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss = torch.sum(self.info_db_loss(db_scores, db_label.float()), dim=-1) * mask

        return torch.mean(info_db_loss)

    def get_total_loss(self, rec_loss, info_db_loss):
        if self.pretrain:
            return info_db_loss
        else:
            return rec_loss + 0.025 * info_db_loss

    def forward(
        self,
        concept_mask,
        seed_sets,
        db_vec,
        labels=None,
        response=None,
        rec=None,
        bsz=None,
    ):
        if bsz == None:
            bsz = len(seed_sets)

        # graph network
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)  # entity_encoder
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)  # word_encoder

        user_representation_list = []
        db_con_mask = []
        for i, seed_set in enumerate(seed_sets):
            if len(seed_set) == 0:
                user_representation_list.append(torch.zeros(self.dim).to(device))
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb = torch.stack(user_representation_list)
        db_con_mask = torch.stack(db_con_mask)

        graph_con_emb = con_nodes_features[concept_mask]
        con_emb_mask = concept_mask == self.concept_padding

        con_user_emb = graph_con_emb
        con_user_emb, attention = self.self_attn(con_user_emb, con_emb_mask.to(device))
        user_emb = self.user_norm(torch.cat([con_user_emb, db_user_emb], dim=-1))
        uc_gate = F.sigmoid(self.gate_norm(user_emb))
        user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        if response is not None:
            info_db_loss = self.infomax_loss(db_nodes_features, con_user_emb, db_vec, db_con_mask)
            rec_loss = self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.to(device))
            rec_loss = torch.sum(rec_loss * rec.float().to(device))

            loss = self.get_total_loss(rec_loss, info_db_loss)
            labels = torch.tensor(labels)
            rec_loss = torch.tensor(rec_loss)
        else:
            loss = None
            rec_loss = None
        result = {"loss": loss, "labels": labels, "rec_logits": entity_scores, "rec_loss": rec_loss}
        return result


class GenModel(nn.Module):
    def __init__(
        self, config, pretrained_word_embedding, dbpedia_edge_sets, concept_edge_sets, mask4key, mask4movie, **kwargs
    ):
        super().__init__()
        self.vocab_size = pretrained_word_embedding.size(0)

        self.batch_size = config.batch_size
        self.max_r_length = config.max_r_length
        self.NULL_IDX = config.padding_idx
        self.END_IDX = config.end_idx
        self.longest_label = config.longest_label
        self.register_buffer("START", torch.LongTensor([config.start_idx]))

        self.pad_idx = config.padding_idx
        self.embeddings = _create_embeddings(
            self.vocab_size, config.embedding_size, self.pad_idx, pretrained_word_embedding
        )

        self.concept_embeddings = _create_entity_embeddings(config.n_concept + 1, config.dim, 0)
        self.concept_padding = 0

        self.n_positions = config.n_positions

        self.encoder = TransformerEncoder(
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            embedding_size=config.embedding_size,
            ffn_size=config.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.embeddings,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=config.relu_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=config.learn_positional_embeddings,
            embeddings_scale=config.embeddings_scale,
            reduction=False,
            n_positions=self.n_positions,
        )

        self.decoder = TransformerDecoderKG(
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            embedding_size=config.embedding_size,
            ffn_size=config.embedding_size,
            vocabulary_size=self.vocab_size,
            embedding=self.embeddings,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=config.relu_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=config.learn_positional_embeddings,
            embeddings_scale=config.embeddings_scale,
            n_positions=int(self.n_positions),
        )

        self.db_norm = nn.Linear(config.dim, config.embedding_size)
        self.kg_norm = nn.Linear(config.dim, config.embedding_size)

        self.db_attn_norm = nn.Linear(config.dim, config.embedding_size)
        self.kg_attn_norm = nn.Linear(config.dim, config.embedding_size)

        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.self_attn = SelfAttentionLayer_batch(config.dim, config.dim)

        self.self_attn_db = SelfAttentionLayer(config.dim, config.dim)

        self.user_norm = nn.Linear(config.dim * 2, config.dim)
        self.gate_norm = nn.Linear(config.dim, 1)
        self.copy_norm = nn.Linear(config.embedding_size * 2 + config.embedding_size, config.embedding_size)
        self.representation_bias = nn.Linear(config.embedding_size, self.vocab_size)

        self.output_en = nn.Linear(config.dim, config.n_entity)

        self.embedding_size = config.embedding_size
        self.dim = config.dim

        self.dbpedia_edge_sets = dbpedia_edge_sets
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(config.n_entity, self.dim, config.n_relation, num_bases=config.num_bases)
        self.concept_edge_sets = concept_edge_sets

        self.concept_GCN = GCNConv(self.dim, self.dim)

        self.mask4key = mask4key
        self.mask4movie = mask4movie
        self.mask4 = self.mask4key + self.mask4movie

        self.pretrain = config.pretrain

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(
        self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen
    ):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """

        xs = self._starts(bsz)
        logits = []
        for i in range(maxlen):
            scores, _ = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db)
            scores = scores[:, -1:, :]
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            db_attn_norm = self.db_attn_norm(attention_db)
            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))
            con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)
            voc_logits = F.linear(scores, self.embeddings.weight)
            sum_logits = voc_logits + con_logits
            _, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)

        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db)  # batch*r_l*hidden

        kg_attention_latent = self.kg_attn_norm(attention_kg)
        db_attention_latent = self.db_attn_norm(attention_db)
        copy_latent = self.copy_norm(
            torch.cat(
                [
                    kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                    db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                    latent,
                ],
                -1,
            )
        )

        con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(
            0
        )  # F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        sum_logits = logits + con_logits
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.to(device), score_view.to(device))
        return loss

    def forward(
        self,
        context,
        concept_mask,
        seed_sets,
        entity_vector,
        response=None,
        entity=None,
        cand_params=None,
        prev_enc=None,
        maxlen=None,
        bsz=None,
    ):

        if response is not None:
            self.longest_label = max(self.longest_label, response.size(1))
        else:
            maxlen = 20

        if bsz == None:
            bsz = len(seed_sets)

        encoder_states = prev_enc if prev_enc is not None else self.encoder(context)

        # graph network---------------------------------------------------------------------------------------------------
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features = self.concept_GCN(self.concept_embeddings.weight, self.concept_edge_sets)

        user_representation_list = []
        db_con_mask = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).to(device))
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb = torch.stack(user_representation_list)
        db_con_mask = torch.stack(db_con_mask)

        graph_con_emb = con_nodes_features[concept_mask]
        con_emb_mask = concept_mask == self.concept_padding

        con_user_emb = graph_con_emb
        con_user_emb, attention = self.self_attn(con_user_emb, con_emb_mask.to(device))

        # generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen = con_nodes_features
        con_emb4gen = con_nodes_features4gen[concept_mask]
        con_mask4gen = concept_mask != self.concept_padding

        kg_encoding = (self.kg_norm(con_emb4gen), con_mask4gen.to(device))

        db_emb4gen = db_nodes_features[entity_vector]
        db_mask4gen = entity_vector != 0

        db_encoding = (self.db_norm(db_emb4gen), db_mask4gen.to(device))

        if response is not None:  # if self.test == False
            scores, preds = self.decode_forced(
                encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, response
            )
            gen_loss = torch.mean(self.compute_loss(scores, response))

        else:
            scores, preds = self.decode_greedy(
                encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, bsz, maxlen or self.longest_label
            )
            gen_loss = torch.tensor(1.0)
        loss = gen_loss

        result = {"loss": loss, "preds": preds, "logits": scores, "response": response, "context": context}
        return result
