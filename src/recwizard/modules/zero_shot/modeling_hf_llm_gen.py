from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer

from recwizard import BaseModule, monitor
from recwizard.modules.zero_shot.configuration_hf_llm_gen import HFLLMGenConfig
from recwizard.utils import create_chat_message

import logging

logger = logging.getLogger(__name__)


class HFLLMGen(BaseModule):
    """
    The generator implemented based on hugggingface LLMs.

    """

    config_class = HFLLMGenConfig
    tokenizer_class = AutoTokenizer

    def __init__(self, config: HFLLMGenConfig, **kwargs):
        """
        Initializes the instance based on the config file.

        Args:
            config (HFLLMGenConfig): The config file.
        """

        super().__init__(config, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

    @monitor
    def response(
        self,
        raw_input,
        tokenizer,
        recs: List[str] = None,
        return_dict=False,
        max_new_tokens=128,
        **kwargs,
    ):
        """
        Generate a template to response the processed user's input.

        Args:
            raw_input (str): The user's raw input.
            tokenizer (BaseTokenizer, optional): A tokenizer to process the raw input.
            recs (list, optional): The recommended movies.
            max_tokens (int): The maximum number of tokens used for ChatLLM API.
            model_name (str, optional): The specified LLM model's name.
            return_dict (bool): Whether to return a dict or a list.
            **kwargs: The other arguments.

        Returns:
            str: The template to response the processed user's input.
        """

        """ LLama format Reference:
            <s>[INST] <<SYS>>
                {{ system_prompt }}
                <</SYS>>
            {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
        """

        # fill movies into the prompt
        prompt = self.config.prompt.copy()
        if recs is not None:
            formatted_movies = ", ".join([f'{i + 1}. "{movie}"' for i, movie in enumerate(recs)])
            prompt["content"] = prompt["content"].format(formatted_movies)

        # raw input to chat message
        chat_message = create_chat_message(raw_input)

        try:  # assume system role is supported
            chat_message.append(prompt)
            chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)
        except:  # change the role to user
            chat_message[-1]["role"] = "user"
            if len(chat_message) > 1 and chat_message[-1]["role"] == "user":  # merge the user's messages
                chat_message[-1]["content"] += " " + chat_message.pop(-1)["content"]
            chat_inputs = tokenizer.apply_chat_template(chat_message, tokenize=False)

        inputs = tokenizer(chat_inputs, return_tensors="pt").to(self.device)

        # model generates
        res = self.model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        # new token ids and related logits
        generated_tokens = res[:, inputs.input_ids.shape[-1] :]
        output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # return the output
        if return_dict:
            return {
                "chat_inputs": chat_inputs,
                "gen_ids": generated_tokens,
                "output": output,
            }
        return output


if __name__ == "__main__":
    # Test the LLMGen module
    model_name = "google/gemma-2b-it"
    config = HFLLMGenConfig(model_name=model_name)
    llm_gen = HFLLMGen(config)

    string = (
        "User: Hello; <sep> System: Hi <sep> User: I am looking for some movies similar to <entity>Titanic</entity>."
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    llm_gen.to("cuda")
    res = llm_gen.response(string, tokenizer, recs=["The Godfather", "Avatar", "Inception"])
    print(res)
