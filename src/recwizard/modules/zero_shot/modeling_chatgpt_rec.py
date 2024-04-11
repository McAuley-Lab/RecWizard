import json
from openai import OpenAI

from recwizard import BaseModule, monitor
from recwizard.modules.zero_shot.tokenizer_chatgpt import ChatGPTTokenizer
from recwizard.modules.zero_shot.configuration_chatgpt_rec import ChatGPTRecConfig


class ChatGPTRec(BaseModule):
    """
    The recommender implemented based on OpanAI's GPT models.

    """

    config_class = ChatGPTRecConfig
    tokenizer_class = ChatGPTTokenizer

    def __init__(self, config: ChatGPTRecConfig, **kwargs):
        """
        Initializes the instance based on the config file.

        Args:
            config (ChatGPTRecConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.
        """

        super().__init__(config, **kwargs)
        self.client = OpenAI()
        self.prompt = config.prompt
        self.backup_prompt = config.backup_prompt

    @classmethod
    def get_tokenizer(cls, **kwargs):
        """
        Get a tokenizer.

        Returns:
            (ChatGPTRecTokenizer): the tokenizer.
        """
        # return lambda x: {'context': x}
        return ChatGPTTokenizer()

    @monitor
    def response(
        self,
        raw_input: str,
        tokenizer=None,
        topk=3,
        max_tokens=None,
        temperature=0.5,
        return_dict=False,
        **kwargs,
    ):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (dict): A dict that contains the question and its related information.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the question.
            topk (int): The number of answers.
            max_tokens (int): The maximum number of tokens used for ChatGPT API.
            temperature (float): The temperature value used for ChatGPT API.
            model_name (str, optional): The specified GPT model's name.
            return_dict (bool): Whether to return a dict or a list.


        Returns:
            list: The answers.
        """
        # raw input to chat message with prompt
        messages = tokenizer(raw_input)["messages"]
        messages.append(self.prompt)

        # call and get the response
        try:
            res = (
                self.client.chat.completions.create(
                    model=self.config.model_name, max_tokens=max_tokens, messages=messages, temperature=temperature
                )
                .choices[0]
                .message.content
            )
            answers = json.loads(res)
            if len(answers) < topk:
                raise ValueError()

            output = [answer["name"] for answer in answers][:topk]
            links = [answer["uri"] for answer in answers][:topk]

        # if the response is not in the expected format, use the backup prompt
        except:
            messages[-1] = self.config.backup_prompt
            res = (
                self.client.chat.completions.create(
                    model=self.config.model_name, max_tokens=max_tokens, messages=messages, temperature=temperature
                )
                .choices[0]
                .message.content
            )
            output = res.split(",")[:topk]
            if len(output) < topk:
                output = output + [""] * (topk - len(output))
            links = [""] * topk

        if return_dict:
            return {"input": messages, "output": output, "links": {name: link for name, link in zip(output, links)}}
        else:
            return output
