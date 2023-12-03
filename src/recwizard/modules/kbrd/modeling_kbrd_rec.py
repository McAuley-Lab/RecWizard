from typing import Union, List
from transformers.utils import ModelOutput

import pickle as pkl
import torch

from recwizard.module_utils import BaseModule
from recwizard.utility import WrapSingleInput
from .configuration_kbrd_rec import KBRDRecConfig
from .tokenizer_kbrd_rec import KBRDRecTokenizer
from .entity_attention_encoder import KBRD

from recwizard.modules.monitor import monitor


class KBRDRec(BaseModule):
    config_class = KBRDRecConfig
    tokenizer_class = KBRDRecTokenizer
    LOAD_SAVE_IGNORES = {"encoder.text_encoder", "decoder"}

    def __init__(
        self,
        config: KBRDRecConfig,
        edge_index=None,
        edge_type=None,
        item_index=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # prepare weights
        edge_index = self.prepare_weight(edge_index, "edge_index")
        edge_type = self.prepare_weight(edge_type, "edge_type")
        item_index = self.prepare_weight(item_index, "item_index")

        # include the item_index into state_dict
        self.item_index = torch.nn.Parameter(item_index, requires_grad=False)

        # then load KBRD recommender
        self.model = KBRD(
            n_entity=config.n_entity,
            n_relation=config.n_relation,
            sub_n_relation=config.sub_n_relation,
            dim=config.dim,
            edge_idx=edge_index,
            edge_type=edge_type,
            num_bases=config.num_bases,
        )

        # set the criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor = None,
    ):
        scores = self.model(input_ids, attention_mask)

        loss = None if labels is None else self.criterion(scores, labels)

        return ModelOutput({"rec_logits": scores, "rec_loss": loss})

    @WrapSingleInput
    @monitor
    def response(
        self,
        raw_input: Union[List[str], str],
        tokenizer: KBRDRecTokenizer,
        return_dict=False,
        topk=3,
    ):
        entities = tokenizer(raw_input)["entities"].to(self.device)
        inputs = {
            "input_ids": entities,
            "attention_mask": entities != tokenizer.pad_entity_id,
        }
        logits = self.forward(**inputs)["rec_logits"]
        # mask all non-movie indices
        offset = 0 * logits.clone() + float("-inf")
        offset[:, self.item_index] = 0
        logits += offset
        movieIds = logits.topk(k=topk, dim=1).indices.tolist()
        output = tokenizer.batch_decode(movieIds)

        if return_dict:
            return {
                "logits": logits,
                "movieIds": movieIds,
                "output": output,
            }
        return output
