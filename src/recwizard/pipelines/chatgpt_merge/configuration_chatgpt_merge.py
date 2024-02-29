from recwizard.utils import START_TAG, END_TAG
from recwizard.configuration_utils import BaseConfig


class ChatgptMergeConfig(BaseConfig):
    """
    The configuration of the CRS pipeline based on OpenAI's GPT models.
    This pipeline is used to merge the results from any recommendation module and generation module.

    Attributes:
        resp_prompt (str): The response prompt.
        model_name(str): The specified GPT model's name.
        prompt(str): The prompt for the GPT model.
        answer_type (str): The type of the answer.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", prompt=None, **kwargs):
        """
        Initializes the instance of this configuration.

        Args:
            model_name (str): The specified GPT model's name.
            prompt (str): The response prompt.
        """

        super().__init__(**kwargs)
        self.model_name = model_name
        self.prompt = (
            prompt
            if prompt is not None
            else "You are a conversational recommender."
            + "This is the conversation between you and the user: {}\n"
            + "This is the generated textual response to the user's input: {}\n"
            + f"And this is the recommendation to the user's input, wrapped by {START_TAG} and {END_TAG}"
            + "{}\n Please based on those information, generate a response containing recommendations to the user."
        )
