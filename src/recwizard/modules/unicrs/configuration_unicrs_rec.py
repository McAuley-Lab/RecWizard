from recwizard.configuration_utils import BaseConfig


class UnicrsRecConfig(BaseConfig):
    def __init__(
        self,
        pretrained_model: str = "",
        kgprompt_config: dict = None,
        num_tokens: int = 0,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.kgprompt_config = kgprompt_config
        self.num_tokens = num_tokens
        self.pad_token_id = pad_token_id
