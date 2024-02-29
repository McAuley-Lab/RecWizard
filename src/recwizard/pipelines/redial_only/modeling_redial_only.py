from recwizard.pipeline_utils import BasePipeline
from recwizard import monitor
from recwizard.pipelines.redial_only.configuration_redial_only import RedialOnlyConfig


class RedialOnlyPipeline(BasePipeline):
    config_class = RedialOnlyConfig

    def __init__(self, config, use_rec_logits=True, **kwargs):
        super().__init__(config, **kwargs)
        self.use_rec_logits = use_rec_logits

    def forward(self, **input):
        recs = self.rec_module.forward(**input)
        return self.gen_module.forward(**input, recs=recs)

    @monitor
    def response(self, query, return_dict=False, rec_args={}, gen_args={}, **kwargs):
        # This pipeline is customized for the ReDial model since it requires the logits from the recommendation module

        # Get the recommendations
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=True, **rec_args)
        rec_logits = rec_output["logits"] if self.use_rec_logits else rec_output["output"]

        # Get the generations
        gen_output = self.gen_module.response(
            query, tokenizer=self.gen_tokenizer, rec_logits=rec_logits, return_dict=return_dict, **gen_args
        )

        # Output
        if return_dict:
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": gen_output["output"],
            }
        return gen_output
