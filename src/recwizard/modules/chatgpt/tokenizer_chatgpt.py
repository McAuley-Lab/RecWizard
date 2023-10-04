from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utility import SEP_TOKEN


class ChatgptTokenizer():
    """
    The tokenizer for the generator based on OpenAI's GPT models.
    """

    def __init__(self, **kwargs):
        """
        Initializes the instance of this tokenizer.
        """

        pass

    def __call__(self, context, **kwargs):
        """
        Process the raw input by extracting the pure text.

        Args:
            context (str): The raw input.

        Returns:
            (dict): A dict that contains the extracted text.
        """

        def preprocess(text):
            text = text.replace('<entity>', '')
            text = text.replace('</entity>', '')
            return text

        texts = preprocess(context).split(SEP_TOKEN)
        messages = []
        user = 'User:'
        system = 'System:'
        for text in texts:
            if text.startswith(user):
                messages.append({'role': 'user', 'content': text[len(user):].strip(' ')})
            elif text.startswith(system):
                messages.append({'role': 'system', 'content': text[len(system):].strip(' ')})
            else:
                messages.append({'role': 'user', 'content': text})

        return {'messages': messages}
