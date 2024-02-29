import torch
from recwizard import BaseModule

from recwizard.modules.kgsf.configuration_kgsf_gen import KGSFGenConfig
from recwizard.modules.kgsf.tokenizer_kgsf_gen import KGSFGenTokenizer

from recwizard.modules.kgsf.original_model import GenModel

from recwizard.utils import create_chat_message
from recwizard import monitor


class KGSFGen(BaseModule):
    config_class = KGSFGenConfig
    tokenizer_class = KGSFGenTokenizer

    def __init__(
        self,
        config,
        pretrained_word_embedding=None,
        dbpedia_edge_sets=None,
        concept_edge_sets=None,
        mask4key=None,
        mask4movie=None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        # prepare weights
        pretrained_word_embedding = self.prepare_weight(pretrained_word_embedding, "pretrained_word_embedding")
        dbpedia_edge_sets = self.prepare_weight(dbpedia_edge_sets, "dbpedia_edge_sets")
        concept_edge_sets = self.prepare_weight(concept_edge_sets, "concept_edge_sets")
        mask4key = self.prepare_weight(mask4key, "mask4key")
        mask4movie = self.prepare_weight(mask4movie, "mask4movie")

        # include the tensors into state_dict
        self.pretrained_word_embedding = torch.nn.Parameter(pretrained_word_embedding, requires_grad=False)
        self.dbpedia_edge_sets = torch.nn.Parameter(dbpedia_edge_sets, requires_grad=False)
        self.concept_edge_sets = torch.nn.Parameter(concept_edge_sets, requires_grad=False)
        self.mask4key = torch.nn.Parameter(mask4key, requires_grad=False)
        self.mask4movie = torch.nn.Parameter(mask4movie, requires_grad=False)

        self.model = GenModel(
            config, pretrained_word_embedding, dbpedia_edge_sets, concept_edge_sets, mask4key, mask4movie, **kwargs
        )

    def generate(self, context_ids, concept_ids, item_ids, entity_ids, *args, **kwargs):
        return self.model(context_ids, concept_ids, item_ids, entity_ids, *args, **kwargs)

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False):

        # Raw input to chat message
        chat_message = create_chat_message(raw_input)
        chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        # Chat message to model inputs
        inputs = tokenizer(chat_inputs, return_token_type_ids=False, return_tensors="pt").to(self.device)
        inputs = {
            "context_ids": inputs["word"]["input_ids"],
            "concept_ids": inputs["concept"]["input_ids"],
            "item_ids": inputs["item"]["input_ids"],
            "entity_ids": inputs["entity"]["input_ids"],
        }

        # Model generates
        outputs = self.generate(**inputs)
        output = tokenizer.decode(outputs["preds"].flatten().tolist(), skip_special_tokens=True)

        # Return the output
        if return_dict:
            return {
                "chat_inputs": chat_inputs,
                "logits": outputs["logits"],
                "gen_ids": outputs["preds"],
                "output": output,
            }
        return output


if __name__ == "__main__":

    path = "../../../../../local_repo/kgsf-gen"
    kgsf_gen = KGSFGen.from_pretrained(path)
    tokenizer = KGSFGenTokenizer.from_pretrained(path)

    resp = kgsf_gen.response(
        raw_input="System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?",
        tokenizer=tokenizer,
    )

    # save raw inputs
    import os
    import numpy as np

    os.makedirs(os.path.join(path, "raw_model_inputs"), exist_ok=True)
    for name, data in [
        ("pretrained_word_embedding", kgsf_gen.pretrained_word_embedding.cpu().numpy()),
        ("dbpedia_edge_sets", kgsf_gen.dbpedia_edge_sets.cpu().numpy()),
        ("concept_edge_sets", kgsf_gen.concept_edge_sets.cpu().numpy()),
        ("mask4key", kgsf_gen.mask4key.cpu().numpy()),
        ("mask4movie", kgsf_gen.mask4movie.cpu().numpy()),
    ]:
        with open(os.path.join(path, "raw_model_inputs", f"{name}.npy"), "wb") as f:
            np.save(f, data)

    print(resp)

    kgsf_gen.save_pretrained(path)
