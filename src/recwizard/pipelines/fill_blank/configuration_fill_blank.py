from recwizard.configuration_utils import BaseConfig


class FillBlankConfig(BaseConfig):
    """
    Configuration class for handling fill-in-the-blank recommendation patterns and response prompts.
    """

    def __init__(self, rec_pattern: str=r"<movie>", resp_prompt='System:', **kwargs):
        """
        Initialize the FillBlankConfig.

        Args:
            rec_pattern (str, optional): The pattern for the potential answers.
                                         Defaults to "<movie>".
            resp_prompt (str, optional): A string representing the response prompt. Defaults to 'System:'.
            **kwargs: Additional keyword arguments to be passed to the BaseConfig constructor.
        """
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt

