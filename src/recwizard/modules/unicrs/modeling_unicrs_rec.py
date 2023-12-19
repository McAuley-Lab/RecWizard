from typing import Union, List
from transformers.utils import ModelOutput
import torch

from recwizard import BaseModule, monitor
from recwizard.utility import WrapSingleInput, deterministic_seed, Singleton, EntityLink
from .tokenizer_unicrs_rec import UnicrsRecTokenizer
from .kg_prompt import KGPrompt
from .prompt_gpt2 import PromptGPT2LMHead
from .configuration_unicrs_rec import UnicrsRecConfig


class UnicrsRec(BaseModule):
    config_class = UnicrsRecConfig
    tokenizer_class = UnicrsRecTokenizer
    LOAD_SAVE_IGNORES = {'encoder.text_encoder', 'decoder'}

    def __init__(self, config: UnicrsRecConfig, edge_index=None, edge_type=None, use_rec_prefix=True,
                 dataset='redial_unicrs', **kwargs):
        super().__init__(config, **kwargs)
        edge_index = self.prepare_weight(edge_index, "edge_index")
        edge_type = self.prepare_weight(edge_type, "edge_type")
        self.encoder = KGPrompt(**config.kgprompt_config, edge_index=edge_index, edge_type=edge_type)
        self.decoder = Singleton('unicrs.PromptGPT2', PromptGPT2LMHead.from_pretrained(config.pretrained_model))
        with deterministic_seed():  # TODO: consider using utils.resize_embeddings
            self.decoder.resize_token_embeddings(config.num_tokens)
        self.decoder.requires_grad_(False)
        self.use_rec_prefix = use_rec_prefix  # set to False in prompt pretrain
        self.entity_linker = EntityLink(dataset=dataset)

    def forward(self, context, prompt, entities, rec_labels=None):
        prompt_embeds, entity_embeds = self.encoder(
            entity_ids=entities,
            prompt=prompt,
            rec_mode=True,
            use_rec_prefix=self.use_rec_prefix
        )
        hidden_states = self.decoder(**context, prompt_embeds=prompt_embeds, conv=False).logits
        input_ids = context.get("input_ids")
        inputs_embeds = context.get("input_embeds")
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if input_ids is not None:
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1

        rec_logits = hidden_states[range(batch_size), sequence_lengths]  # (bs, hidden_size)
        rec_logits @= entity_embeds.T  # (bs, n_item)

        if rec_labels is not None:
            rec_loss = torch.nn.functional.cross_entropy(rec_logits, rec_labels)
        else:
            rec_loss = None

        return ModelOutput({
            "rec_logits": rec_logits,
            "rec_loss": rec_loss
        })

    @WrapSingleInput
    @monitor
    def response(self, raw_input: Union[List[str], str], tokenizer, return_dict=False, topk=3):  # NOTE: unfinished
        inputs = tokenizer(raw_input).to(self.device)
        logits = self.forward(**inputs)['rec_logits']
        movieIds = logits.topk(k=topk, dim=1).indices.tolist()
        output = tokenizer.batch_decode(movieIds)

        if return_dict:
            return {
                'logits': logits,
                'movieIds': movieIds,
                'output': output,
            }
        return output
