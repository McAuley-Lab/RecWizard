import re
from recwizard.model_utils import BasePipeline
from .configuration_chatgpt_agent import ChatgptAgentConfig
import sys
sys.path.append('./src')
import openai


class ChatgptAgent(BasePipeline):
    """
    The CRS model based on OpenAI's GPT models.

    Attributes:
        model (function): The GPT model.
        prompt(str): The prompt for the GPT model.
        movie_pattern (str): The pattern for the potential answers.
        answer_type (str): The type of the answer.
    """

    config_class = ChatgptAgentConfig

    def __init__(self, config, prompt=None, model_name=None, temperature=1, **kwargs):
        """
        Initializes the instance of this CRS model.

        Args:
            config (ChatgptAgentConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name. 
        """

        super().__init__(config, **kwargs)
        self.movie_pattern = re.compile(config.rec_pattern)
        self.answer_type = config.answer_type
        self.model_name = config.model_name if model_name is None else model_name
        self.prompt = config.prompt if prompt is None else prompt
        self.temperature = temperature

    def response(self, query, **kwargs):
        """
        Response to the user's input.

        Args:
            query (str): The user's input.

        Returns:
            str: The processed user's input
            str: The response to the user's input.
        """

        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer)
        resp_start = gen_output.rfind(self.config.resp_prompt)
        context, resp = gen_output[:resp_start], gen_output[resp_start:]
        n_movies = len(self.movie_pattern.findall(resp))
        rec = self.rec_module.response(gen_output, topk=n_movies, tokenizer=self.rec_tokenizer)

        pure_output = gen_output.split('System: ')[1]
        input = ('Please replace the "<{}>" in the text "{}" by the following phrases "{}" respectively. '
                 .format(self.answer_type, pure_output, rec))
        input += self.prompt

        messages = [
            {
                "role": "user",
                "content": input
            }
        ]
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )['choices'][0]['message']['content']

        return resp
