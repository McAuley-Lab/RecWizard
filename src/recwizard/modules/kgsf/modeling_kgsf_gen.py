from typing import Union, List
from transformers.utils import ModelOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import json
import pickle as pkl
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from recwizard import BaseModule

#from recwizard.utility.utils import WrapSingleInput, deterministic_seed, getEntityName2id, getEntity2id

from .configuration_kgsf_gen import KGSFGenConfig
from .tokenizer_kgsf_gen import KGSFGenTokenizer
from .utils import _create_embeddings,_create_entity_embeddings, _edge_list, _concept_edge_list4GCN, seed_everything
from .graph_utils import SelfAttentionLayer,SelfAttentionLayer_batch
from .transformer_utils import TransformerEncoder, TransformerDecoderKG,  _build_decoder4kg
from recwizard.modules.monitor import monitor
class KGSFGen(BaseModule):
    config_class = KGSFGenConfig
    tokenizer_class = KGSFGenTokenizer

    def __init__(self, config, **kwargs):
        """Initialize the KGSFGen module.

        Args:
            config (KGSFGenConfig): The configuration of the KGSFGen module.
        """
        super().__init__(config,**kwargs)
        self.opt = vars(config)  # converting config to dictionary if needed
        dictionary = json.load(open('../kgsfdata/word2index_redial.json', encoding='utf-8'))

        self.batch_size = config.batch_size
        self.max_r_length = config.max_r_length
        self.NULL_IDX = config.padding_idx
        self.END_IDX = config.end_idx
        self.longest_label = config.longest_label
        self.register_buffer('START', torch.LongTensor([config.start_idx]))
        
        self.pad_idx = config.padding_idx
        self.embeddings = _create_embeddings(
            dictionary, config.embedding_size, self.pad_idx
        )

        self.concept_embeddings = _create_entity_embeddings(
            config.n_concept + 1, config.dim, 0)
        self.concept_padding = 0

        self.kg = pkl.load(
            open("../kgsfdata/subkg.pkl", "rb")
        )

        self.n_positions = config.n_positions
        
        self.encoder = TransformerEncoder(
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            embedding_size=config.embedding_size,
            ffn_size=config.ffn_size,
            vocabulary_size=len(dictionary)+4,
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
            vocabulary_size=len(dictionary)+4,
            embedding=self.embeddings,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=config.relu_dropout,
            padding_idx=self.pad_idx,
            learn_positional_embeddings=config.learn_positional_embeddings,
            embeddings_scale=config.embeddings_scale,
            n_positions=int(self.n_positions),
        )
        # self.decoder = _build_decoder4kg(config,dictionary,self.embeddings,self.pad_idx,self.n_positions)
        
            
        
        self.db_norm = nn.Linear(config.dim, config.embedding_size)
        self.kg_norm = nn.Linear(config.dim, config.embedding_size)

        self.db_attn_norm=nn.Linear(config.dim,config.embedding_size)
        self.kg_attn_norm=nn.Linear(config.dim,config.embedding_size)
        
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.self_attn = SelfAttentionLayer_batch(config.dim, config.dim)

        self.self_attn_db = SelfAttentionLayer(config.dim, config.dim)

        self.user_norm = nn.Linear(config.dim * 2, config.dim)
        self.gate_norm = nn.Linear(config.dim, 1)
        self.copy_norm = nn.Linear(config.embedding_size * 2 + config.embedding_size, config.embedding_size)
        self.representation_bias = nn.Linear(config.embedding_size, len(dictionary) + 4)

        self.output_en = nn.Linear(config.dim, config.n_entity)

        self.embedding_size = config.embedding_size
        self.dim = config.dim

        edge_list, self.n_relation = _edge_list(self.kg, config.n_entity, hop=2)
        edge_list = list(set(edge_list))
        print(len(edge_list), self.n_relation)
        self.dbpedia_edge_sets = torch.LongTensor(edge_list).to(device)
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(config.n_entity, self.dim, self.n_relation, num_bases=config.num_bases)
        self.concept_edge_sets = _concept_edge_list4GCN()
        self.concept_GCN = GCNConv(self.dim, self.dim)

        self.mask4key=torch.Tensor(np.load('../kgsfdata/mask4key.npy')).to(device)
        self.mask4movie=torch.Tensor(np.load('../kgsfdata/mask4movie.npy')).to(device)
        self.mask4=self.mask4key+self.mask4movie

        params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                    self.concept_embeddings.parameters(),
                    self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
                    self.gate_norm.parameters(), self.output_en.parameters()]
        for param in params:
            for pa in param:
                pa.requires_grad = False
                
        self.pretrain = config.pretrain
        
    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)
    
    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen):
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
        incr_state = None
        logits = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            # print(scores,scores.shape,incr_state)
            scores = scores[:, -1:, :]
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            
            db_attn_norm = self.db_attn_norm(attention_db)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))

            # logits = self.output(latent)
            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            sum_logits = voc_logits + con_logits #* (1 - gate)
            # print('sum logits ', sum_logits)
            _, preds = sum_logits.max(dim=-1)
            # print('preds ', preds)
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

        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden
        
        print('-------------------------------------')
        with open('mydata.json', 'r') as f:
            # Load the data
            mydata = json.load(f)
        mydata['inputs'] = inputs.cpu().numpy().tolist()
        mydata['encoder_states'] = [tensor.detach().cpu().numpy().tolist() for tensor in encoder_states]
        mydata['encoder_states_kg'] = [tensor.detach().cpu().numpy().tolist() for tensor in encoder_states_kg]
        mydata['encoder_states_db'] = [tensor.detach().cpu().numpy().tolist() for tensor in encoder_states_db]
        with open('mydata.json', 'w') as f:
            json.dump(mydata, f, indent=4)
        
        kg_attention_latent=self.kg_attn_norm(attention_kg)
        db_attention_latent=self.db_attn_norm(attention_db)
        copy_latent=self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))

        con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        sum_logits = logits+con_logits
        _, preds = sum_logits.max(dim=2)
        return logits, preds


    def compute_loss(self, output, scores):
        """Compute the loss for the model's output.

        Args:
            output (torch.Tensor): The output tensor from the model.
            scores (torch.Tensor): The target scores tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.to(device), score_view.to(device))
        return loss
    
    def forward(self, context, response, concept_mask, seed_sets, entity_vector, entity=None, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        """Forward pass of the model.

        Processes the input data through the encoder and graph network, and performs generation based on the context and graph embeddings.

        Args:
            context (torch.Tensor): The context tensor.
            response (torch.Tensor): The response tensor.
            concept_mask (torch.Tensor): The concept mask tensor.
            seed_sets (List[torch.Tensor]): List of entity tensors.
            entity_vector (torch.Tensor): The entity vector tensor.

        Returns:
            dict: A dictionary containing the loss, predictions, response, and context.
        """
        if response is not None:
            self.longest_label = max(self.longest_label, response.size(1))
        else:
            maxlen=20

        if bsz == None:
            bsz = len(seed_sets)

        encoder_states = prev_enc if prev_enc is not None else self.encoder(context)
        
        # graph network---------------------------------------------------------------------------------------------------
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        user_representation_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
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
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda())

        #generation---------------------------------------------------------------------------------------------------
        con_nodes_features4gen=con_nodes_features
        con_emb4gen = con_nodes_features4gen[concept_mask]
        con_mask4gen = concept_mask != self.concept_padding
        
        kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda())

        db_emb4gen=db_nodes_features[entity_vector]
        db_mask4gen=entity_vector!=0
        
        db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda())
    
        
        if response is not None: # if self.test == False
            scores, preds = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, response)
            gen_loss = torch.mean(self.compute_loss(scores, response))

        else:
            
            scores, preds = self.decode_greedy(
                encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
                bsz,
                maxlen or self.longest_label
            )
            gen_loss = torch.tensor(1.0)
        loss = gen_loss

        result = {'loss': loss, 'preds': preds, 'response': response, 'context': context}
        return result
    

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False):
        """Generate a response based on the raw input.

        This function processes the raw input through the tokenizer and the model to generate a textual response.

        Args:
            raw_input (str): The raw input text.
            tokenizer: The tokenizer used for processing the input.
            return_dict (bool, optional): Flag to return the output as a dictionary. Defaults to False.

        Returns:
            str: The generated textual response.
        """
        tokenized_input = tokenizer.encode(raw_input)
        output = self.forward(**tokenized_input)['preds']
        decoded_text = tokenizer.decode(output)
        return decoded_text
