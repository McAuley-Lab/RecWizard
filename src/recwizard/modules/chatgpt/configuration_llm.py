from recwizard.configuration_utils import BaseConfig


class LLMConfig(BaseConfig):
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
        self.model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.max_gen_len = max_gen_len
        self.answer_type = 'movie'
        self.answer_mask = '<movie>'
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = {
                'role': 'system',
                'content': (  
                    ' You are a customer support recommending movies to the user.'
                    ' Our operating platform returns suggested movies in real time from the dialogue history.'
                    ' You may choose from the suggestions and elaborate on why the user may like them.'
                    ' Or you can choose to reply without a recommendation.'
                    ' Now The system is suggesting the following movies: {}.'
                    # ' Carefully review the dialogue history before I write a response to the user.'
                    ' If a movie comes in the format with a year, e.g. ""Inception (2010)"",'
                    ' you should see the year (2010) as a part of the movie name.'
                    ' You should not use the format ""Inception" (2010)" by leaving the year out of the quotation mark.'
                    ' You should keep in mind that the system suggestion is only for reference.'
                    # ' If the user is saying things like thank you or goodbye,'
                    ' You should prioritize giving a quick short response over throwing more movies at the user,'
                    ' especially when the user is likely to be leaving.'
                )
            }
            # self.prompt = {
            #     'role': 'user',
            #     'content': (  
            #         # ' The only answer is {self.answer_name}, do not give other answer.'
            #         # ' Your sentence should be friendly and kind.'
            #         # ' Do not reply me, and directly answer the question.'
            #         # ' Do not say you cannot answer the question. Directly answer it.'
            #         # ' Here is a history of dialogue: "{}",'
            #         # f' and your  {self.answer_mask}.'
            #         f' Now please generate a template for a program to fill. Do not directly give the name of the {self.answer_type}.'
            #         f' Respond with a template where each {self.answer_type} is replaced with a {self.answer_mask} mask.'
            #         # f' I will carefully review the previous dialogue before I respond to the user.'
            #         # f' Due to regulations I am NOT allowed to return any name of {self.answer_type} directly.'
            #         f' You should use a {self.answer_mask} mask to represent the name of the {self.answer_type},'
            #         # f' if there is any {self.answer_type} in the answer.'
            #         f' For example, if you intend to say: "Inception is a good movie".'
            #         f' You should instead say "{self.answer_mask} is a good movie".'
            #         f' Your response may or may not refer to a {self.answer_type}.'
            #         f' Never mention any concrete name of {self.answer_type} in your response!'
            #         # f' But notice that I should not use the form "({self.answer_mask})"'
            #     )
            # }
