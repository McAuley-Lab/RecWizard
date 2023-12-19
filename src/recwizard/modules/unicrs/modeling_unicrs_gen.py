import torch
from transformers import GenerationConfig
from transformers.utils import ModelOutput

from recwizard import BaseModule, monitor
from recwizard.utility import deterministic_seed, SEP_TOKEN, Singleton
from .configuration_unicrs_gen import UnicrsGenConfig
from .tokenizer_unicrs_gen import UnicrsGenTokenizer
from .kg_prompt import KGPrompt
from .prompt_gpt2 import PromptGPT2LMHead


class UnicrsGen(BaseModule):
    config_class = UnicrsGenConfig
    tokenizer_class = UnicrsGenTokenizer
    LOAD_SAVE_IGNORES = {'encoder.text_encoder', 'decoder'}

    def __init__(self, config: UnicrsGenConfig, edge_index=None, edge_type=None, **kwargs):
        super().__init__(config, **kwargs)
        edge_index = self.prepare_weight(edge_index, "edge_index")
        edge_type = self.prepare_weight(edge_type, "edge_type")
        self.encoder = KGPrompt(**config.kgprompt_config, edge_index=edge_index, edge_type=edge_type)
        self.decoder = Singleton('unicrs.PromptGPT2', PromptGPT2LMHead.from_pretrained(config.pretrained_model))
        # self.decoder.config.pad_token_id = config.pad_token_id
        with deterministic_seed():
            self.decoder.resize_token_embeddings(config.num_tokens)
        self.decoder.requires_grad_(False)
        self.max_gen_len = config.max_gen_len

    def forward(self, context, prompt, entities, labels, **kwargs):
        prompt_embeds = self.encoder(
            entity_ids=entities,
            prompt=prompt,
            rec_mode=False,
            use_conv_prefix=True
        )
        output = self.decoder(**context, prompt_embeds=prompt_embeds, labels=labels)
        return ModelOutput({
            "conv_logits": output.logits,
            "conv_loss": output.loss
        })

    @torch.no_grad()
    def generate(self, context, entities, prompt, **kwargs):
        prompt_embeds = self.encoder(
            entity_ids=entities,
            prompt=prompt,
            rec_mode=False,
            use_conv_prefix=True
        )
        return self.decoder.generate(**context,
                                     generation_config=GenerationConfig(pad_token_id=self.config.pad_token_id),
                                     prompt_embeds=prompt_embeds,
                                     max_new_tokens=self.max_gen_len,
                                     no_repeat_ngram_size=3,
                                     )

    @monitor
    def response(self, raw_input: str, tokenizer, return_dict=False):
        raw_input = raw_input.strip(' ') + SEP_TOKEN + 'System:'
        tokenized_input = tokenizer([raw_input]).to(self.device)
        genIds = self.generate(**tokenized_input)
        decoded_text = tokenizer.batch_decode(genIds)[0]
        text = decoded_text.replace(str(tokenizer.tokenizers[0].eos_token), '\n').strip()
        resp_start = text.rfind('System:')
        context, resp = text[:resp_start], text[resp_start:]
        if return_dict:
            return {
                'tokenized_input': tokenized_input,
                'genIds': genIds,
                'decoded_text': decoded_text,
                'context': context,
                'output': resp
            }
        return resp
