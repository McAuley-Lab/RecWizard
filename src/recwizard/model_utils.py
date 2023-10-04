import logging
from typing import Iterable

from .modules.monitor import monitor
from .configuration_utils import BaseConfig
from .module_utils import BaseModule
from .tokenizer_utils import BaseTokenizer


class BasePipeline(BaseModule):
    """
    The base model for recwizard.
    """
    config_class = BaseConfig

    def __init__(self,
                 config: BaseConfig = None,
                 rec_module: BaseModule = None,
                 gen_module: BaseModule = None,
                 rec_tokenizer: BaseTokenizer = None,
                 gen_tokenizer: BaseModule = None,
                 **kwargs):
        """
        Initialize the model with the recommender/generator modules and the corresponding tokenizers.

        If the tokenizers are not passed, the model will try to get the tokenizers by calling `get_tokenizer` on the modules.

        Args:
            config (BaseConfig): The config file as is used in `transformers.PreTrainedModel`.
            rec_module (BaseModule): The recommender module.
            rec_tokenizer (BaseTokenizer): The tokenizer for the recommender module.
            gen_module (BaseModule): The generator module.
            gen_tokenizer (BaseTokenizer): The tokenizer for the generator module.
            **kwargs: The other keyword arguments used for `transformers.PreTrainedModel`.
        """
        super().__init__(config, **kwargs)
        self.rec_module = rec_module
        self.gen_module = gen_module
        # Intitialize the LOAD_SAVE_IGNORES, for saving the model with minimal space
        if rec_module and isinstance(rec_module.LOAD_SAVE_IGNORES, Iterable):
            for prefix in list(self.rec_module.LOAD_SAVE_IGNORES):
                self.LOAD_SAVE_IGNORES.add(f'rec_module.{prefix}')
        if gen_module and isinstance(gen_module.LOAD_SAVE_IGNORES, Iterable):
            for prefix in list(self.gen_module.LOAD_SAVE_IGNORES):
                self.LOAD_SAVE_IGNORES.add(f'gen_module.{prefix}')
        # Initialize the tokenizers if no tokenizer is passed
        self.rec_tokenizer = rec_tokenizer or self.rec_module.get_tokenizer()
        self.gen_tokenizer = gen_tokenizer or self.gen_module.get_tokenizer()
        for tokenizer_name in ['rec_tokenizer', 'gen_tokenizer']:
            if self.__dict__[tokenizer_name] is None:
                logging.warning(
                    f'{self.__class__} cannot initialize the {tokenizer_name}. You may want to pass the tokenizer manually')

    @monitor
    def response(self, query: str, return_dict: bool = False, rec_args: dict = None, gen_args: dict = None, *args,
                 **kwargs):
        r"""
        The main function for the model to generate a response given a query.

        Args:
            query (str): The formatted dialogue history. See the format at :doc:`/quick_start/example`.
            return_dict (bool): if set to True, will return a dict of outputs instead of a single output.
            rec_args (dict): The arguments passed to the recommender module.
            gen_args (dict): The arguments passed to the generator module.

        """
        rec_args = rec_args or {}
        gen_args = gen_args or {}
        if not return_dict:
            raise NotImplementedError
        return {
            'rec_output': self.rec_module.response(query, self.rec_tokenizer, **rec_args),
            'gen_output': self.gen_module.response(query, self.gen_tokenizer, **gen_args)
        }

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
