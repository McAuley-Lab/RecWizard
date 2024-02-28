import re

from recwizard.pipeline_utils import BasePipeline
from recwizard.utils import SEP_TOKEN, EntityLink, monitor

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
        self.entity_linker = EntityLink()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query, return_dict=False, gen_args=None, rec_args=None):
        gen_args = gen_args or {}
        rec_args = rec_args or {}
        # generate response template
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer, return_dict=return_dict, **gen_args)
        resp = gen_output["output"] if return_dict else gen_output
        # resp = 'System: <movie>, <movie>, and <movie> are some good options.'
        # generate topk recommendations
        k_movies = len(self.movie_pattern.findall(resp))
        if k_movies > 0:
            rec_input = query
            if self.use_resp:
                rec_input += SEP_TOKEN + resp
            rec_output = self.rec_module.response(
                rec_input,
                topk=k_movies,
                tokenizer=self.rec_tokenizer,
                return_dict=return_dict,
                **rec_args,
            )
            rec = rec_output["output"] if return_dict else rec_output

            # replace <movie> placeholder in response
            for i in range(k_movies):
                resp = self.movie_pattern.sub(rec[i], resp, count=1)
        else:
            rec_input = {}
            rec_output = {}
        if return_dict:
            if rec_output.get("links"):  # special case for llm
                item_links = rec_output["links"]
            elif "item_ids" in rec_output:
                item_ids = rec_output.get("item_ids")
                item_names = self.rec_tokenizer.batch_decode(item_ids)
                item_links = {item_name: self.entity_linker(item_name) for item_name in item_names}
            else:
                item_links = {}
            return {
                "gen_input": query,
                "gen_output": gen_output,
                "rec_input": rec_input,
                "rec_output": rec_output,
                "output": resp,
                "links": item_links,
            }
        return resp
