from typing import Union, List
from transformers.utils import ModelOutput

import torch

from recwizard.module_utils import BaseModule
from recwizard.utility import WrapSingleInput
from .configuration_kbrd_rec import KBRDRecConfig
from .tokenizer_kbrd_rec import KBRDRecTokenizer
from .entity_attention_encoder import KBRD

from recwizard.modules.monitor import monitor


class KBRDRec(BaseModule):
    """KBRDRec is a module that implements the KBRD recommender."""

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
        """Initialize the KBRDRec module.

        Args:
            config (KBRDRecConfig): The configuration of the KBRDRec module.
            edge_index (torch.LongTensor): The edges (node pairs) of the knowledge graph in KBRD, shape: (2, num_edges).
            edge_type (torch.LongTensor): The edge type of the knowledge graph in KBRD, shape: (num_edges,).
            item_index (torch.LongTensor): The recommendation item indices used in KBRD model, shape: (num_items,).
        """

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
        """Forward pass of the KBRDRec module.
        
        Args:
            input_ids (torch.LongTensor): The input ids of the input conversation contexts, shape: (batch_size, seq_len).
            attention_mask (torch.BoolTensor): The attention mask of the input, shape: (batch_size, seq_len).
            labels (torch.LongTensor): The labels of the converastions, optional.
        
        Returns:
            ModelOutput: The output of the model, containing the logits and the loss.
        """

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
        """Generate the response given the input_ids.

        Args:
            raw_input (Union[List[str], str]): The input conversation contexts.
            tokenizer (KBRDRecTokenizer): The tokenizer of the model.
            return_dict (bool): Whether to return the output as a dictionary.
            topk (int): The number of movies to recommend.
        
        Returns:
            Union[List[str], dict]: The dictionary of the model output with `logits`, 
                `movieIds` and textual response if `return_dict` is `True`, else the textual 
                model response only.
        """
        
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
