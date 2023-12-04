from recwizard.configuration_utils import BaseConfig


class SwitchDecodeConfig(BaseConfig):
    """
        Configuration class for controlling decoding behavior with switching mechanism.
    """

    def __init__(self, hidden_size=256, context_size=256, max_seq_length=40, rec_pattern: str=r"<movie>", resp_prompt='System:', **kwargs):
        """
        Initialize the SwitchDecodeConfig.

        Args:
            hidden_size (int, optional): The size of the hidden state in the model. Defaults to 256.
            context_size (int, optional): The size of the context vector in the model. Defaults to 256.
            max_seq_length (int, optional): The maximum sequence length for decoding. Defaults to 40.
            rec_pattern (str, optional): A string representing the recommendation pattern to be used.
                                         Defaults to "<movie>".
            resp_prompt (str, optional): A string representing the response prompt. Defaults to 'System:'.
            **kwargs: Additional keyword arguments to be passed to the BaseConfig constructor.
        """
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.max_seq_length = max_seq_length

