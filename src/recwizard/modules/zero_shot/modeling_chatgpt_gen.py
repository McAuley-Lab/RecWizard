from typing import List
from openai import OpenAI

from recwizard import BaseModule, monitor
from recwizard.modules.zero_shot.configuration_chatgpt_gen import ChatGPTGenConfig
from recwizard.modules.zero_shot.tokenizer_chatgpt import ChatGPTTokenizer


class ChatGPTGen(BaseModule):
    """
    The generator implemented based on OpanAI's GPT models.

    """

    config_class = ChatGPTGenConfig
    tokenizer_class = ChatGPTTokenizer

    def __init__(self, config: ChatGPTGenConfig, **kwargs):
        """
        Initializes the instance based on the ChatGPT configuration.

        Args:
            config (ChatGPTGenConfig): The ChatGPT config for the generator.
        """

        super().__init__(config, **kwargs)
        self.client = OpenAI()
        self.prompt = config.prompt

    @classmethod
    def get_tokenizer(cls, **kwargs):
        """
        Get a tokenizer.

        Returns:
            (ChatGPTTokenizer): the tokenizer.
        """
        # return lambda x: {'context': x}
        return ChatGPTTokenizer()

    @monitor
    def response(
        self,
        raw_input,
        tokenizer,
        recs: List[str] = None,
        max_tokens=None,
        temperature=0.5,
        return_dict=False,
        **kwargs,
    ):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (str): The user's raw input.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the raw input.
            recs (list, optional): The recommended movies.
            max_tokens (int): The maximum number of tokens used for ChatGPT API.
            temperature (float): The temperature value used for ChatGPT API.
            return_dict (bool): Whether to return a dict or a list.

        Returns:
            str: The template to response the processed user's input.
        """

        messages = tokenizer(raw_input)["messages"]
        # Add prompt at end
        prompt = self.prompt.copy()
        if recs is not None:
            formatted_movies = ", ".join([f'{i + 1}. "{movie}"' for i, movie in enumerate(recs)])
            prompt["content"] = prompt["content"].format(formatted_movies)
        messages.append(prompt)

        res = (
            self.client.chat.completions.create(
                model=self.config.model_name,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
            )
            .choices[0]
            .message.content
        )

        res_out = "System: {}".format(res)
        if return_dict:
            return {
                "chat_inputs": messages,
                "output": res_out,
            }
        return res_out


if __name__ == "__main__":
    # Test the ChatGPTGen module
    config = ChatGPTGenConfig(model_name="gpt-3.5-turbo")
    chatgpt_gen = ChatGPTGen(config)

    string = "User: Hello; <sep> System: Hi <sep> User: How are you? <sep> System: I am fine."

    tokenizer = ChatGPTTokenizer()
    res = chatgpt_gen.response(string, tokenizer)
    print(res)
