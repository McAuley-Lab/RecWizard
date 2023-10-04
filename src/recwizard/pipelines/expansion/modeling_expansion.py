from recwizard.model_utils import BasePipeline
from recwizard.modules.monitor import monitor
from .configuration_expansion import ExpansionConfig
from ...utility import EntityLink


class ExpansionPipeline(BasePipeline):
    config_class = ExpansionConfig

    def __init__(self, config, use_rec_logits=True, **kwargs):
        super().__init__(config, **kwargs)
        self.use_rec_logits = use_rec_logits
        self.entity_linker = EntityLink()


    @monitor
    def response(self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        rec_args = rec_args or {}
        gen_args = gen_args or {}
        rec_output = self.rec_module.response(query,
                                              tokenizer=self.rec_tokenizer,
                                              return_dict=True,
                                              **rec_args)
        recs = rec_output['logits'] if self.use_rec_logits else rec_output['output']
        gen_output = self.gen_module.response(query,
                                              tokenizer=self.gen_tokenizer,
                                              recs=recs,
                                              return_dict=return_dict,
                                              **gen_args)
        if return_dict:
            movieIds = gen_output.get('movieIds') or rec_output.get('movieIds')
            movieNames = self.rec_tokenizer.batch_decode(movieIds)
            movieLinks = {movieName: self.entity_linker(movieName) for movieName in movieNames}
            return {
                'rec_output': rec_output,
                'gen_output': gen_output,
                'output': gen_output['output'],
                'links': movieLinks
            }
        return gen_output

    def forward(self, **input):
        recs = self.rec_module.forward(**input)
        return self.gen_module.forward(**input, recs=recs)
