from recwizard.configuration_utils import BaseConfig


class ChatgptRecConfig(BaseConfig):
    """
    The configuration of the recommender based on OpenAI's GPT models.

    Attributes:
        answer_name (str): The special string representing the answer in the response template.
        answer_type (str): The type of the answer.
        prompt(str): The prompt for the GPT model.
        model_name(str): The specified GPT model's name.
    """

    def __init__(self, **kwargs):
        """
        Initializes the instance of this configuration.
        """

        super().__init__(**kwargs)

        # self.answer_name = 'AAAAAAAAAA'
        self.model_name = 'gpt-4'
        self.answer_type = 'movie'
        self.answer_mask = '<movie>'
        self.prompt = {
            'role': 'user',
            'content': (
                # ' I am a movie recommender responsible for filling the given template'
                # ' What phrase could best replace "{}"?'
                # ' The response should be only a name of a {}.'
                # ' You should only response the phrase, and do not say anything else.'
                # ' Your response should contain no more than 8 words.'
                f' It look like your previous response is a a template with some {self.answer_mask} mask.'
                f' Now please fills in with actual names of {self.answer_type}.'
                f' Your answer should be formatted as a json object with a link to the movies wiki page '
                f' so that I could directly parse it.'
                f' For example if there are two {self.answer_mask} in the template,'
                f' and you want to fill them with "Inception" and "The Matrix",'
                f' then your answer should be formatted as follows:'
                ' [{"name": "Inception", "uri": "https://en.wikipedia.org/wiki/Inception"},'
                ' {"name": "The Matrix", "uri": "https://en.wikipedia.org/wiki/The_Matrix"}]'
                f' The returned object should has the same number of {self.answer_type} as in your previous response.'
                f' Do not include anything outside the json object.'
            )
        }
        self.backup_prompt = {
            'role': 'user',
            'content': (
                f' It look like the previous response is a a template with some {self.answer_mask} mask.'
                f' Now please fills in with actual names of {self.answer_type}.'
                f' Your answer should be formatted as a string separated by comma.'
                f' For example if there are three {self.answer_mask} in the template,'
                f' and you want to fill them with "Inception", "The Matrix", and "The Dark Knight",'
                f' then your answer should be formatted as:'
                f' """Inception,The Matrix,The Dark Knight"""'
                f' Do not include anything except the formatted string.'
            )
        }
