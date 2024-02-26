from recwizard.pipeline_utils import BasePipeline
from recwizard import monitor

from .configuration_trivial import TrivialConfig


class TrivialPipeline(BasePipeline):
    config_class = TrivialConfig

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query=str, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        rec_args = rec_args or {}
        gen_args = gen_args or {}
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=return_dict, **rec_args)
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer, return_dict=return_dict, **gen_args)

        if return_dict:
            return {
                "rec": rec_output,
                "gen": gen_output,
            }

        else:
            return f"Generator: {gen_output}\nRecommender: {rec_output}"
