from recwizard.model_utils import BasePipeline
from recwizard.modules.monitor import monitor
from .configuration_expansion import ExpansionConfig
from ...utility import EntityLink


class ExpansionPipeline(BasePipeline):
    """
    A pipeline for expansion tasks using recommendation and generation modules.
    """
    config_class = ExpansionConfig

    def __init__(self, config, use_rec_logits=True, **kwargs):
        """
        Initialize the ExpansionPipeline.

        Args:
            config (ExpansionConfig): An instance of ExpansionConfig containing pipeline configuration.
            use_rec_logits (bool, optional): Whether to use recommendation module logits. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to the BasePipeline constructor.
        """
        super().__init__(config, **kwargs)
        self.use_rec_logits = use_rec_logits
        self.entity_linker = EntityLink()


    @monitor
    def response(self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        """
        Response to the user's input.

        Args:
            query (str): The user's input.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to False.
            rec_args (dict, optional): Additional arguments for the recommendation module. Defaults to None.
            gen_args (dict, optional): Additional arguments for the generation module. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict or Any: If return_dict is True, returns a dictionary with various outputs including
                         'rec_output', 'gen_output', 'output', and 'links'. Otherwise, returns the generated response.
        """
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
        """
        Forward the input through the pipeline modules.

        Args:
            **input: Keyword arguments containing the input data.

        Returns:
            Any: The forward pass result.
        """
        recs = self.rec_module.forward(**input)
        return self.gen_module.forward(**input, recs=recs)
