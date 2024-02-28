from typing import Union, List
from transformers.utils import ModelOutput
import torch

from recwizard import BaseModule, monitor
from recwizard.utils import Singleton, EntityLink, create_chat_message

from recwizard.modules.unicrs.tokenizer_unicrs_rec import UnicrsRecTokenizer
from recwizard.modules.unicrs.original_kg_prompt import KGPrompt
from recwizard.modules.unicrs.original_prompt_gpt2 import PromptGPT2LMHead
from recwizard.modules.unicrs.configuration_unicrs_rec import UnicrsRecConfig


class UnicrsRec(BaseModule):
    config_class = UnicrsRecConfig
    tokenizer_class = UnicrsRecTokenizer
    LOAD_SAVE_IGNORES = {"encoder.text_encoder", "decoder"}

    def __init__(
        self,
        config: UnicrsRecConfig,
        edge_index=None,
        edge_type=None,
        use_rec_prefix=True,
        dataset="redial_unicrs",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        edge_index = self.prepare_weight(edge_index, "edge_index")
        edge_type = self.prepare_weight(edge_type, "edge_type")
        self.encoder = KGPrompt(**config.kgprompt_config, edge_index=edge_index, edge_type=edge_type)
        self.decoder = Singleton("unicrs.PromptGPT2", PromptGPT2LMHead.from_pretrained(config.pretrained_model))
        self.decoder.resize_token_embeddings(config.num_tokens)
        self.decoder.requires_grad_(False)
        self.use_rec_prefix = use_rec_prefix
        self.entity_linker = EntityLink()

    def forward(self, context, prompt, entities, rec_labels=None):
        prompt_embeds, entity_embeds = self.encoder(
            entity_ids=entities, prompt=prompt, rec_mode=True, use_rec_prefix=self.use_rec_prefix
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

        return ModelOutput({"rec_logits": rec_logits, "rec_loss": rec_loss})

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False, topk=3):

        # raw input to chat message
        chat_message = create_chat_message(raw_input)
        chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        # chat message to model inputs
        inputs = tokenizer(chat_inputs, return_tensors="pt", return_token_type_ids=False).to(self.device)
        inputs = {"context": inputs["context"], "prompt": inputs["prompt"], "entities": inputs["item"]["input_ids"]}
        logits = self.forward(**inputs)["rec_logits"]

        # get topk items
        item_ids = logits.topk(k=topk, dim=1).indices.flatten().tolist()
        output = tokenizer.decode(item_ids)

        # return the output
        if return_dict:
            return {
                "chat_inputs": chat_inputs,
                "logits": logits,
                "item_ids": item_ids,
                "output": output,
            }
        return output


if __name__ == "__main__":
    path = "../../../../../local_repo/unicrs-rec"
    tokenizer = UnicrsRecTokenizer.from_pretrained(path)
    unicrs_rec = UnicrsRec.from_pretrained(path)

    print(
        unicrs_rec.response(
            "System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?",
            tokenizer=tokenizer,
            return_dict=True,
        )
    )

    unicrs_rec.save_pretrained(path)

    # test load
    unicrs_rec = UnicrsRec.from_pretrained(path)
    print(
        unicrs_rec.response(
            "System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?",
            tokenizer=tokenizer,
            return_dict=True,
        )
    )

    # save raw model inputs
    import os
    import numpy as np

    os.makedirs(os.path.join(path, "raw_model_inputs"), exist_ok=True)
    for name, data in [
        ("edge_index", unicrs_rec.encoder.edge_index.cpu().numpy()),
        ("edge_type", unicrs_rec.encoder.edge_type.cpu().numpy()),
    ]:
        with open(os.path.join(path, "raw_model_inputs", f"{name}.npy"), "wb") as f:
            np.save(f, data)
