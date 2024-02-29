from recwizard.pipeline_utils import BasePipeline
from recwizard import monitor

from recwizard.utils import create_rec_list
from recwizard.pipelines.trivial.configuration_trivial import TrivialConfig


class TrivialPipeline(BasePipeline):
    config_class = TrivialConfig

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query=str, return_dict=False, rec_args={}, gen_args={}, **kwargs):
        # Get the recommendations
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=return_dict, **rec_args)

        # Get the generation
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer, return_dict=return_dict, **gen_args)

        # Output
        if return_dict:
            output = f"Generator: {gen_output['output']}\nRecommender: {rec_output['output']}"
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": output,
            }

        else:
            return f"Generator: {gen_output}\nRecommender: {rec_output}"
