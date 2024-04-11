""" This is not a "real" tokenizer since the chatGPT tokenization is not public.

    We use this class to make the code more modular.
"""

from recwizard.utils import create_chat_message, START_TAG, END_TAG


class ChatGPTTokenizer:
    """
    The tokenizer for the generator based on OpenAI's GPT models.
    """

    def __call__(self, context, **kwargs):
        """
        Process the raw input by extracting the pure text.

        Args:
            context (str): The raw input.

        Returns:
            (dict): A dict that contains the extracted text.
        """

        def preprocess(text):
            text = text.replace(START_TAG, "")
            text = text.replace(END_TAG, "")
            return text

        texts = preprocess(context)
        messages = create_chat_message(texts)

        return {"messages": messages}
