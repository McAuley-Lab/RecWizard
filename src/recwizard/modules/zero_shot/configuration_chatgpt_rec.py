from recwizard.configuration_utils import BaseConfig


class ChatGPTRecConfig(BaseConfig):
    """
    The configuration of the recommender based on OpenAI's GPT models.

    Attributes:
        model_name(str): The specified GPT model's name.
        max_gen_len (int): The maximum length to set in the generator.
        answer_name (str): The special string used to represent the answer in the response.
        answer_mask (str): The type of the answer.
        prompt(str): The prompt for the GPT model.
        backup_prompt(str): The backup prompt for the GPT model when the primary prompt failed.
    """

    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        max_gen_len: int = 0,
        answer_type="movie",
        answer_mask="<movie>",
        prompt: dict = None,
        backup_prompt=None,
        **kwargs,
    ):
        """
        Initializes the instance of this configuration.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_gen_len = max_gen_len
        self.answer_type = answer_type
        self.answer_mask = answer_mask
        self.prompt = prompt or {
            "role": "system",
            "content": (
                f" Your previous response is a a template with some {self.answer_mask} mask."
                f" Now please fill in with actual names of {self.answer_type}."
                f" Your answer should be formatted as a json object with a link to the movies wiki page "
                f" so that it could be directly parsed."
                f" For example if there are two {self.answer_mask} in the template,"
                f' and you want to fill them with "Inception" and "The Matrix",'
                f" then your answer should be formatted as follows:"
                ' [{"name": "Inception", "uri": "https://en.wikipedia.org/wiki/Inception"},'
                ' {"name": "The Matrix", "uri": "https://en.wikipedia.org/wiki/The_Matrix"}]'
                f" The returned object should has the same number of {self.answer_type} as in your previous response."
                f" Do not include anything outside the json object."
            ),
        }
        self.backup_prompt = backup_prompt or {
            "role": "system",
            "content": (
                f" Your previous response is a a template with some {self.answer_mask} mask."
                f" Now please fill in with actual names of {self.answer_type}."
                f" Your answer should be formatted as a string separated by comma."
                f" For example if there are three {self.answer_mask} in the template,"
                f' and you want to fill them with "Inception", "The Matrix", and "The Dark Knight",'
                f" then your answer should be formatted as:"
                f' """Inception,The Matrix,The Dark Knight"""'
                f" Do not include anything except the formatted string."
            ),
        }
