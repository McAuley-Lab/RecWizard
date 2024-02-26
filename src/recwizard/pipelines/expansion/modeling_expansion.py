from recwizard.pipeline_utils import BasePipeline
from recwizard import monitor
from recwizard.utility import EntityLink
from .configuration_expansion import ExpansionConfig


class ExpansionPipeline(BasePipeline):
    config_class = ExpansionConfig

    def __init__(self, config, use_rec_logits=True, **kwargs):
        super().__init__(config, **kwargs)
        self.use_rec_logits = use_rec_logits
        self.entity_linker = EntityLink()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError

    @monitor
    def response(self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        rec_args = rec_args or {}
        gen_args = gen_args or {}
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=True, **rec_args)
        recs = rec_output["logits"] if self.use_rec_logits else rec_output["output"]
        gen_output = self.gen_module.response(
            query, tokenizer=self.gen_tokenizer, recs=recs, return_dict=return_dict, **gen_args
        )
        if return_dict:
            item_ids = gen_output.get("item_ids") or rec_output.get("item_ids")
            item_names = self.rec_tokenizer.batch_decode(item_ids)
            item_links = {item_name: self.entity_linker(item_name) for item_name in item_names}
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": gen_output["output"],
                "links": item_links,
            }
        return gen_output

    def forward(self, **input):
        recs = self.rec_module.forward(**input)
        return self.gen_module.forward(**input, recs=recs)
