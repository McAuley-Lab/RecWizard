from recwizard.configuration_utils import BaseConfig


class ExpansionConfig(BaseConfig):
    """
    Configuration class for handling expansion patterns and response prompts.

    Args:
        rec_pattern (str, optional): A string representing the recommendation pattern to be used.
                                     Defaults to "<movie>".
        resp_prompt (str, optional): A string representing the response prompt. Defaults to 'System:'.
        **kwargs: Additional keyword arguments to be passed to the BaseConfig constructor.
    """
    def __init__(self, rec_pattern: str=r"<movie>", resp_prompt='System:', **kwargs):
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt

