from recwizard.configuration_utils import BaseConfig


class SwitchDecodeConfig(BaseConfig):

    def __init__(self, hidden_size=256, context_size=256, max_seq_length=40, rec_pattern: str=r"<movie>", resp_prompt='System:', **kwargs):
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.max_seq_length = max_seq_length

