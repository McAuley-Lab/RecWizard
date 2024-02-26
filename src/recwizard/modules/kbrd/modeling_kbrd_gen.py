import torch

from recwizard import monitor
from recwizard.module_utils import BaseModule
from recwizard.modules.kbrd.configuration_kbrd_gen import KBRDGenConfig
from recwizard.modules.kbrd.tokenizer_kbrd_gen import KBRDGenTokenizer
from recwizard.modules.kbrd.modeling_kbrd_rec import KBRDRecConfig, KBRDRec
from recwizard.modules.kbrd.original_transformer_encoder_decoder import TransformerGeneratorModel

from recwizard.utility import create_chat_message


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
            entities (torch.LongTensor): The item-related entities tagged in the input, shape: (batch_size, entity_max_num).
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
            user_representation, _ = self.model.kbrd.model.user_representation(entities, entity_attention_mask)
            self.model.user_representation = user_representation.detach()

        return self.model(*(input_ids,), bsz=bsz, ys=labels, maxlen=maxlen)

    @monitor
    def response(
        self,
        raw_input: str,
        tokenizer: KBRDGenTokenizer,
        return_dict=False,
    ):
        """Generate the response given the raw input.

        Args:
            raw_input str: The raw input of the conversation contexts.
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
        # raw input to chat message
        chat_message = create_chat_message(raw_input)
        chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        # chat message to model inputs
        results = tokenizer(chat_inputs, return_token_type_ids=False, return_tensors="pt").to(self.device)
        inputs = {
            "input_ids": results["nltk"]["input_ids"],
            "entities": results["entity"]["input_ids"],
            "entity_attention_mask": results["entity"]["attention_mask"],
        }

        # model generates
        outputs = self.generate(**inputs)
        output = tokenizer.decode(outputs[1].flatten().tolist(), skip_special_tokens=True)

        # return the output
        if return_dict:
            return {"chat_inputs": chat_inputs, "logits": outputs[0], "gen_ids": outputs[1], "output": output}

        return output


if __name__ == "__main__":

    # load kbrd generator
    path = "../../../../../local_repo/kbrd-gen"
    kbrd_gen = KBRDGen.from_pretrained(path)

    tokenizer = KBRDGenTokenizer.from_pretrained(path)
    resp = kbrd_gen.response(
        raw_input="I like <entity>Avatar</entity>, and you?",
        tokenizer=tokenizer,
    )

    kbrd_gen.save_pretrained(path)

    print(resp)
