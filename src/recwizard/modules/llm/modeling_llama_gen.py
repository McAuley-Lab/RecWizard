import os
from typing import Union, List
import openai
from transformers import AutoModelForCausalLM


from recwizard import BaseModule
from recwizard.modules.monitor import monitor
from recwizard.utility import DeviceManager
from .configuration_llm import LLMConfig
from .tokenizer_llama import LlamaTokenizer
import logging

logger = logging.getLogger(__name__)


class LlamaGen(BaseModule):
    """
    The generator implemented based on OpanAI's GPT models.

    """

    config_class = LLMConfig
    tokenizer_class = LlamaTokenizer

    def __init__(
        self, config: LLMConfig, prompt=None, model_name=None, debug=False, **kwargs
    ):
        """
        Initializes the instance based on the config file.

        Args:
            config (ChatgptGenConfig): The config file.
            prompt (str, optional): A prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.
        """

        super().__init__(config, **kwargs)
        model_name = model_name or config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompt = config.prompt if prompt is None else prompt
        self.debug = debug

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, config=None, prompt=None, model_name=None
    ):
        """
        Get an instance of this class.

        Args:
            config:
            pretrained_model_name_or_path:
            prompt (str, optional): The prompt to override the prompt from config file.
            model_name (str, optional): The specified GPT model's name.

        Returns:
             the instance.
        """
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        return cls(config, prompt=prompt, model_name=model_name)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        self.config.save_pretrained(
            save_directory=save_directory, push_to_hub=push_to_hub
        )

    @classmethod
    def get_tokenizer(cls, **kwargs):
        """
        Get a tokenizer.

        Returns:
            (ChatgptTokenizer): the tokenizer.
        """
        # return lambda x: {'context': x}
        return LlamaTokenizer()

    @monitor
    def response(
        self,
        raw_input,
        tokenizer,
        recs: List[str] = None,
        max_tokens=None,
        temperature=0.5,
        model_name=None,
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
            model_name (str, optional): The specified GPT model's name.
            return_dict (bool): Whether to return a dict or a list.

        Returns:
            str: The template to response the processed user's input.
        """

        """ LLama format Reference:
            <s>[INST] <<SYS>>
                {{ system_prompt }}
                <</SYS>>
            {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
        """

        # Add prompt at end
        # suppose the last input is the user's input
        prompt = self.prompt.copy()
        if recs is not None:
            formatted_movies = ", ".join(
                [f'{i + 1}. "{movie}"' for i, movie in enumerate(recs)]
            )
            prompt["content"] = prompt["content"].format(formatted_movies)
        raw_input += f"<<SYS>>{prompt['content']}<</SYS>>"

        encodings = tokenizer(raw_input, return_tensors="pt", padding=True)
        encodings = DeviceManager.copy_to_device(encodings, self.model.device)

        res = self.model.generate(
            **encodings,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded_text = tokenizer.decode(res[0], skip_special_tokens=True)
        resp_start = decoded_text.rfind("[/INST]") + len("[/INST]")
        resp = decoded_text[resp_start:].strip(" ")
        output = "System: {}".format(resp)
        if return_dict:
            return {
                "input": raw_input,
                "output": output,
            }
        return output
