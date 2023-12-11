from typing import Union, List

import torch

from recwizard.module_utils import BaseModule
from .configuration_kbrd_gen import KBRDGenConfig
from .tokenizer_kbrd_gen import KBRDGenTokenizer
from .modeling_kbrd_rec import KBRDRecConfig, KBRDRec
from .transformer_encoder_decoder import TransformerGeneratorModel

from recwizard.modules.monitor import monitor


class KBRDGen(BaseModule):
    """KBRDGen is a module that combines KBRDRec and TransformerGeneratorModel."""

    config_class = KBRDGenConfig
    tokenizer_class = KBRDGenTokenizer
    LOAD_SAVE_IGNORES = {"encoder.text_encoder", "decoder"}

    def __init__(self, config: KBRDGenConfig, **kwargs):
        """Initialize the KBRDGen module.

        Args:
            config (KBRDGenConfig): The configuration of the KBRDGen module.
        """

        super().__init__(config, **kwargs)

        self.model = TransformerGeneratorModel(
            config,
            kbrd_rec=KBRDRec(config=KBRDRecConfig.from_dict(config.rec_module_config)),
        )

    def generate(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
        entities: torch.LongTensor = None,
        entity_attention_mask: torch.BoolTensor = None,
        bsz: int = None,
        maxlen: int = 64,
    ):
        """Generate the response given the input_ids.

        Args:
            input_ids (torch.LongTensor): The input ids of the input conversation contexts, shape: (batch_size, seq_len).
            labels (torch.LongTensor, optional): The labels of the converastions.
            entities (torch.LongTensor): The movie-related entities tagged in the input, shape: (batch_size, entity_max_num).
            entity_attention_mask (torch.BoolTensor): The entity attention mask of the input, `True` if the token is an entity, shape: (batch_size, entity_max_num).
            bsz (int, optional): The batch size of the input, can be inferred from the input_ids.
            maxlen (int): The maximum length of the input.

        Returns:
            ModelOutput: The output of the model.

        Examples:
            ```python
                # load kbrd generator
                kbrd_gen = KBRDGen.from_pretrained("recwizard/kbrd-gen-redial")

                # test model generate
                kbrd_gen = kbrd_gen
                input_ids = torch.LongTensor([[0] * 55])
                entities, entity_attention_mask = (
                    torch.LongTensor([[]]),
                    torch.BoolTensor([[]]),
                )

                kbrd_gen.generate(
                    input_ids=input_ids,
                    labels=None,
                    entities=entities,
                    entity_attention_mask=entity_attention_mask,
                )
            ```
        """

        if bsz is None:
            bsz = input_ids.size(0)

        if entities is not None:
            user_representation, _ = self.model.kbrd.model.user_representation(
                entities, entity_attention_mask
            )
            self.model.user_representation = user_representation.detach()

        return self.model(*(input_ids,), bsz=bsz, ys=labels, maxlen=maxlen)

    def response(
        self,
        raw_input: Union[List[str], str],
        tokenizer: KBRDGenTokenizer,
        return_dict=False,
    ):
        """Generate the response given the raw input.

        Args:
            raw_input (Union[List[str], str]): The raw input of the conversation contexts.
            tokenizer (KBRDGenTokenizer): The tokenizer of the model.
            return_dict (bool): Whether to return the output as a dictionary (with logits, pred_ids and textual response).

        Returns:
            Union[List[str], dict]: The dictionary of the model output with `logits`, 
                `pred_ids` and textual response if `return_dict` is `True`, else the textual 
                model response only.

        Examples:
            ```python
                # load kbrd generator
                kbrd_gen = KBRDGen.from_pretrained("recwizard/kbrd-gen-redial")

                tokenizer = KBRDGenTokenizer.from_pretrained("recwizard/kbrd-gen-redial")
                kbrd_gen.response(
                    raw_input=["I like <entity>Titanic</entity>! Can you recommend me more?"],
                    tokenizer=tokenizer,
                )
            ```
        """
        input_ids = torch.LongTensor(tokenizer(raw_input)["input_ids"]).to(self.device)
        entities = (
            torch.LongTensor(tokenizer.tokenizers[-1](raw_input)["entities"])
            .to(self.device)
            .unsqueeze(dim=0)
        )
        inputs = {
            "input_ids": input_ids,
            "entities": entities,
            "entity_attention_mask": entities != tokenizer.tokenizers[-1].pad_entity_id,
        }
        outputs = self.generate(**inputs)

        # mask all non-movie indices
        if return_dict:
            return {
                "logits": outputs[0],
                "pred_ids": outputs[1],
                "output": tokenizer.batch_decode(outputs[1].tolist()),
            }

        return tokenizer.batch_decode(outputs[1].tolist())
