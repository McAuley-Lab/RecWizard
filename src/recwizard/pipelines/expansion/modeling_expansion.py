from recwizard.pipeline_utils import BasePipeline
from recwizard import monitor
from recwizard.utils import create_item_list
from recwizard.pipelines.expansion.configuration_expansion import ExpansionConfig


class ExpansionPipeline(BasePipeline):
    config_class = ExpansionConfig

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query, return_dict=False, rec_args={}, gen_args={}, **kwargs):

        # Get the recommendations
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=return_dict, **rec_args)
        rec_list = create_item_list(rec_output["output"] if return_dict else rec_output)

        # Get the generation
        gen_output = self.gen_module.response(
            query, tokenizer=self.gen_tokenizer, recs=rec_list, return_dict=return_dict, **gen_args
        )

        # Output
        output = gen_output
        if return_dict:
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": output,
            }
        return output
