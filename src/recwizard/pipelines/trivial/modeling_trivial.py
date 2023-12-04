from recwizard.model_utils import BasePipeline
from recwizard.modules.monitor import monitor

from .configuration_trivial import TrivialConfig

class TrivialPipeline(BasePipeline):
    """
    A pipeline for generating responses using a trivial model.

    Args:
        config (TrivialConfig): An instance of TrivialConfig containing pipeline configuration.
        **kwargs: Additional keyword arguments to be passed to the BasePipeline constructor.
    """
    config_class = TrivialConfig

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        This method is not implemented in the TrivialPipeline and should be overridden in a subclass.

        Args:
            input_ids: Input IDs for the model.
            attention_mask: Attention mask for the model.
            labels: Labels for the model.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError


    @monitor
    def response(self, query, return_dict=False, rec_args=None, gen_args=None, **kwargs):
        """
        Generate a response using the TrivialPipeline.

        Args:
            query (str): The input query for generating a response.
            return_dict (bool, optional): Whether to return the result as a dictionary. Defaults to False.
            rec_args (dict, optional): Additional arguments for the recommendation module. Defaults to None.
            gen_args (dict, optional): Additional arguments for the generation module. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict or str: If return_dict is True, returns a dictionary with 'rec_logits', 'gen_logits',
                         'rec_output', and 'gen_output'. Otherwise, returns the generated response as a string.
        """
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