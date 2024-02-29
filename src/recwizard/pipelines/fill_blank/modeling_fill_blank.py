import re

from recwizard import monitor
from recwizard.pipeline_utils import BasePipeline
from recwizard.utils import SEP_TOKEN, ASSISTANT_TOKEN, create_item_list

from .configuration_fill_blank import FillBlankConfig


class FillBlankPipeline(BasePipeline):
    config_class = FillBlankConfig

    def __init__(self, config, use_resp=True, **kwargs):
        """Initialize the fill-blank pipeline.

        Args:
            config (FillBlankConfig): The configuration of the fill-blank pipeline.
            use_resp (bool, optional): Whether to use the response as the input for recommendation. Defaults to True.
        """
        super().__init__(config, **kwargs)
        self.movie_pattern = re.compile(config.rec_pattern)
        self.use_resp = use_resp

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query, return_dict=False, gen_args={}, rec_args={}):
        # Get the generation with blanks to fill
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer, return_dict=return_dict, **gen_args)

        # Example: gen_template = '<movie>, <movie>, and <movie> are some good options.', which will be filled.
        gen_template = gen_output["output"] if return_dict else gen_output
        gen_template = "<movie>, <movie>, and <movie> are some good options."

        # Get the recommendations if needed
        rec_input, rec_output = {}, {}
        topk = len(self.movie_pattern.findall(gen_template))
        if topk > 0:
            rec_input = query
            if self.use_resp:
                rec_input += SEP_TOKEN + ASSISTANT_TOKEN + gen_template
            rec_output = self.rec_module.response(
                rec_input,
                topk=topk,
                tokenizer=self.rec_tokenizer,
                return_dict=return_dict,
                **rec_args,
            )
            rec_list = create_item_list(rec_output["output"] if return_dict else rec_output)

            # Replace <movie> placeholder in response
            for i in range(topk):
                gen_template = self.movie_pattern.sub(rec_list[i], gen_template, count=1)

        # Output
        output = gen_template

        if return_dict:
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": output,
            }
        return output
