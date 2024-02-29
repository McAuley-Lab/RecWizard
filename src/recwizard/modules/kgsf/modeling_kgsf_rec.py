import torch
from recwizard import BaseModule

from recwizard.modules.kgsf.tokenizer_kgsf_rec import KGSFRecTokenizer
from recwizard.modules.kgsf.configuration_kgsf_rec import KGSFRecConfig
from recwizard.modules.kgsf.original_model import RecModel

from recwizard import monitor
from recwizard.utils import create_chat_message


class KGSFRec(BaseModule):
    config_class = KGSFRecConfig
    tokenizer_class = KGSFRecTokenizer
    LOAD_SAVE_IGNORES = {"START", "decoder"}

    def __init__(
        self,
        config,
        item_index=None,
        pretrained_word_embedding=None,
        dbpedia_edge_sets=None,
        concept_edge_sets=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # prepare weights
        item_index = self.prepare_weight(item_index, "item_index")
        pretrained_word_embedding = self.prepare_weight(pretrained_word_embedding, "pretrained_word_embedding")
        dbpedia_edge_sets = self.prepare_weight(dbpedia_edge_sets, "dbpedia_edge_sets")
        concept_edge_sets = self.prepare_weight(concept_edge_sets, "concept_edge_sets")

        # include the tensors into state_dict
        self.item_index = torch.nn.Parameter(item_index, requires_grad=False)
        self.pretrained_word_embedding = torch.nn.Parameter(pretrained_word_embedding, requires_grad=False)
        self.dbpedia_edge_sets = torch.nn.Parameter(dbpedia_edge_sets, requires_grad=False)
        self.concept_edge_sets = torch.nn.Parameter(concept_edge_sets, requires_grad=False)

        # initialize the model
        self.model = RecModel(config, pretrained_word_embedding, dbpedia_edge_sets, concept_edge_sets, **kwargs)

    def forward(self, concept_ids, item_ids, entity_ids, *args, **kwargs):
        return self.model(concept_ids, item_ids, entity_ids, *args, **kwargs)

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False, topk=3):

        # Raw input to chat message
        chat_message = create_chat_message(raw_input)
        chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        # Chat message to model inputs
        inputs = tokenizer(chat_inputs, return_token_type_ids=False, return_tensors="pt").to(self.device)
        inputs = {
            "concept_ids": inputs["concept"]["input_ids"],
            "item_ids": inputs["item"]["input_ids"],
            "entity_ids": inputs["entity"]["input_ids"],
        }

        logits = self.forward(**inputs)["rec_logits"]

        # Mask all non-item indices
        offset = 0 * logits.clone() + float("-inf")
        offset[:, self.item_index] = 0
        logits += offset

        # Get topk items
        item_ids = logits.topk(k=topk, dim=1).indices.flatten().tolist()
        output = tokenizer.decode(item_ids)

        # Return the output
        if return_dict:
            return {
                "chat_inputs": chat_inputs,
                "logits": logits,
                "item_ids": item_ids,
                "output": output,
            }
        return output


if __name__ == "__main__":

    path = "../../../../../local_repo/kgsf-rec"
    kgsf_rec = KGSFRec.from_pretrained(path)
    tokenizer = KGSFRecTokenizer.from_pretrained(path)

    resp = kgsf_rec.response(
        raw_input="System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?",
        tokenizer=tokenizer,
        return_dict=True,
    )

    # save raw inputs
    import os
    import numpy as np

    os.makedirs(os.path.join(path, "raw_model_inputs"), exist_ok=True)
    for name, data in [
        ("item_index", kgsf_rec.item_index.cpu().numpy()),
        ("pretrained_word_embedding", kgsf_rec.pretrained_word_embedding.cpu().numpy()),
        ("dbpedia_edge_sets", kgsf_rec.dbpedia_edge_sets.cpu().numpy()),
        ("concept_edge_sets", kgsf_rec.concept_edge_sets.cpu().numpy()),
    ]:
        with open(os.path.join(path, "raw_model_inputs", f"{name}.npy"), "wb") as f:
            np.save(f, data)

    print(resp)

    kgsf_rec.save_pretrained(path)
