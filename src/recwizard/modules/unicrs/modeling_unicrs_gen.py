import torch
from transformers import GenerationConfig
from transformers.utils import ModelOutput

from recwizard import BaseModule, monitor
from recwizard.utils import deterministic_seed, Singleton, create_chat_message

from recwizard.modules.unicrs.configuration_unicrs_gen import UnicrsGenConfig
from recwizard.modules.unicrs.tokenizer_unicrs_gen import UnicrsGenTokenizer
from recwizard.modules.unicrs.original_kg_prompt import KGPrompt
from recwizard.modules.unicrs.original_prompt_gpt2 import PromptGPT2LMHead


class UnicrsGen(BaseModule):
    config_class = UnicrsGenConfig
    tokenizer_class = UnicrsGenTokenizer
    LOAD_SAVE_IGNORES = {"encoder.text_encoder", "decoder"}

    def __init__(self, config: UnicrsGenConfig, edge_index=None, edge_type=None, **kwargs):
        super().__init__(config, **kwargs)
        edge_index = self.prepare_weight(edge_index, "edge_index")
        edge_type = self.prepare_weight(edge_type, "edge_type")
        self.encoder = KGPrompt(**config.kgprompt_config, edge_index=edge_index, edge_type=edge_type)
        self.decoder = Singleton("unicrs.PromptGPT2", PromptGPT2LMHead.from_pretrained(config.pretrained_model))
        # self.decoder.config.pad_token_id = config.pad_token_id
        with deterministic_seed():
            self.decoder.resize_token_embeddings(config.num_tokens)
        self.decoder.requires_grad_(False)
        self.max_gen_len = config.max_gen_len

    def forward(self, context, prompt, entities, labels, **kwargs):
        prompt_embeds = self.encoder(entity_ids=entities, prompt=prompt, rec_mode=False, use_conv_prefix=True)
        output = self.decoder(**context, prompt_embeds=prompt_embeds, labels=labels)
        return ModelOutput({"conv_logits": output.logits, "conv_loss": output.loss})

    @torch.no_grad()
    def generate(self, context, entities, prompt, **kwargs):
        prompt_embeds = self.encoder(entity_ids=entities, prompt=prompt, rec_mode=False, use_conv_prefix=True)
        return self.decoder.generate(
            **context,
            generation_config=GenerationConfig(pad_token_id=self.config.pad_token_id),
            prompt_embeds=prompt_embeds,
            max_new_tokens=self.max_gen_len,
            no_repeat_ngram_size=3,
        )

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False):

        # raw input to chat message
        chat_message = create_chat_message(raw_input)
        chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        # chat message to model inputs
        inputs = tokenizer(chat_inputs, return_tensors="pt", return_token_type_ids=False).to(self.device)
        inputs = {"context": inputs["context"], "prompt": inputs["prompt"], "entities": inputs["item"]["input_ids"]}

        # model generates
        gen_ids = self.generate(**inputs)
        delta_length = gen_ids.shape[-1] - inputs["context"]["input_ids"].shape[-1]
        gen_ids = gen_ids.flatten()[-delta_length:]
        decoded_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # return the output
        if return_dict:
            return {
                "chat_inputs": chat_inputs,
                "gen_ids": gen_ids,
                "output": decoded_text,
            }
        return decoded_text


if __name__ == "__main__":
    path = "../../../../../local_repo/unicrs-gen"
    tokenizer = UnicrsGenTokenizer.from_pretrained(path)
    unicrs_gen = UnicrsGen.from_pretrained(path)

    print(
        unicrs_gen.response(
            "System: Hi! <sep> User: I like <entity>Titanic</entity>! Can you recommend me more?",
            tokenizer=tokenizer,
            return_dict=True,
        )
    )

    unicrs_gen.save_pretrained(path)

    # test load
    unicrs_gen = UnicrsGen.from_pretrained(path)
    print(
        unicrs_gen.response(
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
        ("edge_index", unicrs_gen.encoder.edge_index.cpu().numpy()),
        ("edge_type", unicrs_gen.encoder.edge_type.cpu().numpy()),
    ]:
        with open(os.path.join(path, "raw_model_inputs", f"{name}.npy"), "wb") as f:
            np.save(f, data)
