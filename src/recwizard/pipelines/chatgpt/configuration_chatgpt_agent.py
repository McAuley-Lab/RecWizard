from recwizard.configuration_utils import BaseConfig


class ChatgptAgentConfig(BaseConfig):
    """
    The configuration of the CRS model based on OpenAI's GPT models.

    Attributes:
        rec_pattern (str): The pattern for the potential answers.
        resp_prompt (str): The response prompt.
        model_name(str): The specified GPT model's name. 
        prompt(str): The prompt for the GPT model.
        answer_type (str): The type of the answer.
    """

    def __init__(self, rec_pattern: str=r"<movie>", resp_prompt='System:', **kwargs):
        """
        Initializes the instance of this configuration.

        Args:
            rec_pattern (str): The pattern for the potential answers.
            resp_prompt (str): The response prompt.
        """

        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
        self.resp_prompt = resp_prompt
        self.model_name = 'gpt-4'
        self.prompt = ''
        self.answer_type = 'movie'