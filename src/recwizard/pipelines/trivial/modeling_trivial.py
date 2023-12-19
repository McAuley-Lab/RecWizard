from recwizard.model_utils import BasePipeline
from recwizard import monitor

from .configuration_trivial import TrivialConfig

class TrivialPipeline(BasePipeline):
    config_class = TrivialConfig

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        raise NotImplementedError


    @monitor
    def response(self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        rec_args = rec_args or {}
        gen_args = gen_args or {}
        rec_output = self.rec_module.response(query,
                                              tokenizer=self.rec_tokenizer,
                                              return_dict=True,
                                              **rec_args)
        gen_output = self.gen_module.response(query,
                                              tokenizer=self.gen_tokenizer,
                                              return_dict=True,
                                              **gen_args)
        if return_dict:
            return {
                'rec_logits': rec_output['logits'],
                'gen_logits': gen_output['logits'],
                'rec_output': rec_output['output'],
                'gen_output': gen_output['output']
            }
        
        return gen_output['output'][0] + "\n - " + "\n - ".join(rec_output['output'])