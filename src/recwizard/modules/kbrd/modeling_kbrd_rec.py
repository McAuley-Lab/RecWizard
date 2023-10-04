from typing import Union, List
from transformers.utils import ModelOutput

import pickle as pkl
import torch

from recwizard.module_utils import BaseModule
from recwizard.utility import WrapSingleInput, deterministic_seed, Singleton, EntityLink
from .configuration_kbrd_rec import KBRDRecConfig
from .tokenizer_kbrd_rec import KBRDRecTokenizer
from .shared_encoder import KBRD, _edge_list

class KBRDRec(BaseModule):
    config_class = KBRDRecConfig
    tokenizer_class = KBRDRecTokenizer
    LOAD_SAVE_IGNORES = {'encoder.text_encoder', 'decoder'}

    def __init__(self, config: KBRDRecConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # try to load kg first # TODO: make it similar to UniCRS
        kg = pkl.load(open(config.kg_path, "redial", "subkg.pkl"), "rb")
        edge_list, self.n_relation = _edge_list(kg, config.n_entity)
        edge_list = list(set(edge_list))
        edge_list_tensor = torch.LongTensor(edge_list).cuda()

        # then load KBRD recommender
        self.model = KBRD(
            n_entity=config.n_entity,
            n_relation=config.n_relation,
            dim=config.dim,
            edge_list_tensor=edge_list_tensor,
            num_bases=config.num_bases
        )

    def forward(self, seed_sets: list, labels: torch.LongTensor = None):
        outputs = self.model(seed_sets, labels)
        # TODO: make outputs different when labels differ
