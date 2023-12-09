from recwizard.configuration_utils import BaseConfig


class FillBlankConfig(BaseConfig):
    def __init__(self, rec_pattern: str = r"<movie>", resp_prompt="System:", **kwargs):
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt
