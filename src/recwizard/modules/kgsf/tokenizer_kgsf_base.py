""" Build an cumstomized base Tokenizer, to match the original implementation of KGSF to sentence tokenization and encoding, the example is:

    Therefore, we build a tokenizer, where:
        1. Normalization: Nothing
        2. Pre-Tokenize: Tokenize sentences using Regex
        3. Model: WordLevel mapping from the entities to the indexes
        4. Encoding: Return the indexes of the tokens
        5. Decoding: Return the sentence
"""

from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, Tokenizer, NormalizedString, PreTokenizedString

from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import WordLevel

from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace as DummyPreTokenizer

from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast

from recwizard.utils import DEFAULT_CHAT_TEMPLATE


class KGSFBasePreTokenizer:
    def __init__(self):
        import re

        self.pattern = re.compile(r"<entity>.*?<\/entity>|\w+|[.,!?;]")

    def pre_tokenize(self, pretok: PreTokenizedString) -> List[NormalizedString]:
        return pretok.split(self.kgsf_base_split)

    def kgsf_base_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """Return the entity tokens within the entity tags"""
        text = str(normalized_string)
        splits = []
        for match in self.pattern.finditer(text):
            start, stop = match.start(0), match.end(0)
            splits.append(normalized_string[start:stop])
        return splits


class KGSFBaseBackendTokenizer(BaseTokenizer):

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
            "model": "KGSFBaseTokenizer",
            "unk_token": str(unk_token),
        }

        super().__init__(tokenizer, parameters)


class KGSFBaseTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        lowercase: bool = False,
        chat_template: Optional[str] = DEFAULT_CHAT_TEMPLATE,
        unk_token: Union[str, AddedToken] = "[UNK]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        *args,
        **kwargs,
    ):

        # Initialize the backend entity tokenizer
        tokenizer = KGSFBaseBackendTokenizer(vocab, unk_token=str(unk_token))

        # Initialize the PreTrainedTokenizerFast
        super().__init__(tokenizer_object=tokenizer, *args, **kwargs)

        # Add the custom normalizer, pre-tokenizer and decoder
        if lowercase:
            self.backend_tokenizer.normalizer = Lowercase()
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(KGSFBasePreTokenizer())

        # Protect the vocab tokens
        self.add_tokens(list(vocab.keys()))

        # Save init_inputs
        self.init_inputs = (
            vocab,
            lowercase,
        )

        self.unk_token = unk_token
        self.pad_token = pad_token

        # Set the chat template
        if chat_template is not None:
            self.chat_template = chat_template

    def save_pretrained(self, *args, **kwargs) -> Tuple[str]:
        # Save the tokenizer with the dummy pre-tokenizer and decoder
        # To avoid the error: "TypeError: cannot pickle custom objects"

        self.backend_tokenizer.pre_tokenizer = DummyPreTokenizer()
        return super().save_pretrained(*args, **kwargs)
