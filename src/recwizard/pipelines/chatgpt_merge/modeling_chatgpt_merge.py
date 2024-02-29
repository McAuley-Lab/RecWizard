from recwizard.pipeline_utils import BasePipeline
from recwizard.pipelines.chatgpt_merge.configuration_chatgpt_merge import ChatgptMergeConfig

from openai import OpenAI


class ChatgptMergePipeline(BasePipeline):
    """
    The configuration of the CRS pipeline based on OpenAI's GPT models.
    This pipeline is used to merge the results from any recommendation module and generation module.

    Attributes:
        model (function): The GPT model.
        prompt(str): The prompt for the GPT model.
        movie_pattern (str): The pattern for the potential answers.
        answer_type (str): The type of the answer.
    """

    config_class = ChatgptMergeConfig

    def __init__(self, config: ChatgptMergeConfig, **kwargs):
        """
        Initializes the instance based on the ChatGPT configuration.

        Args:
            config (ChatgptMergeConfig): The ChatGPT config for the generator.
        """

        super().__init__(config, **kwargs)
        self.client = OpenAI()
        self.prompt = config.prompt
        self.model_name = config.model_name

    def response(self, query, return_dict=False, rec_args={}, gen_args={}, **kwargs):
        """
        Response to the user's input.

        Args:
            query (str): The user's input.

        Returns:
            str: The processed user's input
            str: The response to the user's input.
        """

        # Get the recommendations
        rec_output = self.rec_module.response(query, tokenizer=self.rec_tokenizer, return_dict=return_dict, **rec_args)

        # Get the generation
        gen_output = self.gen_module.response(query, tokenizer=self.gen_tokenizer, return_dict=return_dict, **gen_args)

        # Merge the outputs with ChatGPT (define better prompts and related formats yourself)
        prompt = self.config.prompt.format(
            query,
            gen_output["output"] if return_dict else gen_output,
            rec_output["output"] if return_dict else rec_output,
        )
        messages = [{"role": "user", "content": prompt}]

        # Get the merged response
        output = (
            self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
            .choices[0]
            .message.content
        )

        # Output
        if return_dict:
            return {
                "rec_output": rec_output,
                "gen_output": gen_output,
                "output": output,
            }
        return output
