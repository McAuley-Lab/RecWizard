from recwizard.configuration_utils import BaseConfig


class ChatgptGenConfig(BaseConfig):
    """
    The configuration of the generator based on OpenAI's GPT models.

    Attributes:
        answer_name (str): The special string used to represent the answer in the response.
        answer_mask (str): The type of the answer.
        prompt(str): The prompt for the GPT model.
        model_name(str): The specified GPT model's name. 
    """

    def __init__(self, max_gen_len: int = 0, prompt: dict = None, **kwargs):
        """
        Initializes the instance of this configuration.

        Args:
            max_gen_len (int, optional): The maximum length to set in the generator.
        """

        super().__init__(**kwargs)
        self.model_name = 'gpt-4'
        self.max_gen_len = max_gen_len
        # self.answer_name = 'AAAAAAAAAA'
        self.answer_type = 'movie'
        self.answer_mask = '<movie>'
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = {
                'role': 'assistant',
                'content': (  # prompt as assistant
                    # ' The only answer is {self.answer_name}, do not give other answer.'
                    # ' Your sentence should be friendly and kind.'
                    # ' Do not reply me, and directly answer the question.'
                    # ' Do not say you cannot answer the question. Directly answer it.'
                    # ' Here is a history of dialogue: "{}",'
                    # f' and your  {self.answer_mask}.'
                    f' I am generating templates for a program to fill. So I will not directly give the name of the {self.answer_type}.'
                    f' But I will respond with a template where each {self.answer_type} is replaced with a {self.answer_mask} mask.'
                    # f' I will carefully review the previous dialogue before I respond to the user.'
                    # f' Due to regulations I am NOT allowed to return any name of {self.answer_type} directly.'
                    f' I must use a {self.answer_mask} mask to represent the name of the {self.answer_type},'
                    # f' if there is any {self.answer_type} in the answer.'
                    f' For example, when I intend to say: "Inception is a good movie".'
                    f' I should instead say "{self.answer_mask} is a good movie".'
                    f' My response may or may not refer to a {self.answer_type}.'
                    f' But I should never mention any concrete name of {self.answer_type} in the answer!'
                    # f' But notice that I should not use the form "({self.answer_mask})"'
                )
            }
