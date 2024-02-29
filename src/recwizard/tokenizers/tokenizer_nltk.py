""" Build an Entity Tokenizer, where this tokenizer can extract multiple items from sentence like 
    "I like <entity> Titanic (1997) </entity>,  <entity> Affective Neuroscience (2004) </entity>.", the example is:

    - tokenizer = NLTKTokenizer.from_file("vocab.json")
    - tokenizer.encode("I like <entity>Titanic_(1997)</entity>, <entity>Affective_Neuroscience_(2004)</entity>.")
        - Normalization (Remove tags and lowercase): "i like titanic_(1997), affective_neuroscience_(2004)"
        - Pre-Tokenize (NLTK): ["i", "like", "titanic_(1997)</entity>", "affective_neuroscience_(2004)", "."]
        - Model (WorldLevel Mapping): ["i" -> 1, "like" -> 2, ... "." -> 10]
        - Encoding: [1, 2, ... 10]
        - Decoding: "i like titanic_(1997), affective_neuroscience_(2004)"
"""

from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, Tokenizer, NormalizedString, PreTokenizedString

from tokenizers.normalizers import Replace, Sequence, Lowercase
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import WordLevel

from tokenizers.pre_tokenizers import Whitespace as DummyPreTokenizer

from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast

from recwizard.utils import DEFAULT_CHAT_TEMPLATE, START_TAG, END_TAG

import nltk


class NLTKPreTokenizer:
    def __init__(self, language="english", word_tokenizer=None):

        # Initialize the NLTK word tokenizer
        if word_tokenizer == "treebank":
            self.word_tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        else:
            self.word_tokenizer = nltk.NLTKWordTokenizer()

        # Initialize the NLTK sentence tokenizer
        st_path = f"tokenizers/punkt/{language}.pickle"
        try:
            self.sent_tokenizer = nltk.data.load(st_path)
        except LookupError:
            nltk.download("punkt")
            self.sent_tokenizer = nltk.data.load(st_path)

    def pre_tokenize(self, pretok: PreTokenizedString) -> List[NormalizedString]:
        return pretok.split(self.nltk_split)

    def nltk_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """Split a string into sentences using NLTK."""
        sentences = [normalized_string[s:t] for s, t in self.sent_tokenizer.span_tokenize(str(normalized_string))]
        tokenized = []
        for sentence in sentences:
            tokenized += [normalized_string[s:t] for s, t in self.word_tokenizer.span_tokenize(str(sentence))]
        return tokenized


class NLTKBackendTokenizer(BaseTokenizer):

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordLevel(vocab, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordLevel(unk_token=str(unk_token)))

        parameters = {
            "model": "EntityTokenizer",
            "unk_token": str(unk_token),
        }

        super().__init__(tokenizer, parameters)


class NLTKTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        nltk_sent_language: str = "english",
        nltk_word_tokenizer: str = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        chat_template: str = DEFAULT_CHAT_TEMPLATE,
        *args,
        **kwargs,
    ):

        # Initialize the backend entity tokenizer
        tokenizer = NLTKBackendTokenizer(vocab, unk_token=str(unk_token))

        # Initialize the PreTrainedTokenizerFast
        super().__init__(tokenizer_object=tokenizer, *args, **kwargs)

        # Add the custom normalizer, pre-tokenizer and decoder
        self.backend_tokenizer.normalizer = Sequence([Replace(START_TAG, " "), Replace(END_TAG, " "), Lowercase()])
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(
            NLTKPreTokenizer(nltk_sent_language, nltk_word_tokenizer)
        )

        # Protect the vocab tokens
        self.add_tokens(list(vocab.keys()))
        self.unk_token = unk_token

        # Save init_inputs
        self.init_inputs = (vocab, nltk_sent_language, nltk_word_tokenizer)

        # Set the chat template
        if chat_template is not None:
            self.chat_template = chat_template

    def save_pretrained(self, *args, **kwargs) -> Tuple[str]:
        # Save the tokenizer with the dummy pre-tokenizer and decoder
        # To avoid the error: "TypeError: cannot pickle custom objects"

        self.backend_tokenizer.pre_tokenizer = DummyPreTokenizer()
        return super().save_pretrained(*args, **kwargs)


if __name__ == "__main__":
    import os, json
    from recwizard.utils import create_chat_message

    word2id = json.load(open("../../../../local_repo/kbrd-gen/raw_vocab/word2id.json"))

    tokenizer = NLTKTokenizer(vocab=word2id, unk_token="[UNK]", pad_token="[PAD]")

    tokenizer.save_pretrained("test")

    tokenizer = NLTKTokenizer.from_pretrained("test")

    chat = create_chat_message("User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!")

    print(tokenizer.apply_chat_template(chat, tokenize=False))

    print(tokenizer.apply_chat_template(chat, tokenize=True))
