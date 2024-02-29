""" Build an Entity Tokenizer, where this tokenizer can extract multiple items from sentence like 
    "I like <entity> Titanic (1997) </entity>,  <entity> Affective Neuroscience (2004) </entity>.", the example is:

    - tokenizer = EntityTokenizer.from_file("vocab.json")
    - tokenizer.encode("I like <entity>Titanic_(1997)</entity>, <entity>Affective_Neuroscience_(2004)</entity>.")
        - Normalization (None): "I like <entity>Titanic_(1997)</entity>, <entity>Affective_Neuroscience_(2004)</entity>."
        - Pre-Tokenize: ["<entity>Titanic_(1997)</entity>", "<entity>Affective_Neuroscience_(2004)</entity>"]
        - Model (WorldLevel Mapping): ["<entity>Titanic_(1997)</entity>" -> 1, "<entity>Affective_Neuroscience_(2004)</entity>" -> 2]
        - Encoding: [1, 2]
        - Decoding: "<entity>Titanic_(1997)</entity> <entity>Affective_Neuroscience_(2004)</entity>"
"""

from typing import Dict, List, Optional, Tuple, Union
from tokenizers import AddedToken, Tokenizer, NormalizedString, PreTokenizedString

from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import WordLevel

from tokenizers.pre_tokenizers import Whitespace as DummyPreTokenizer
from tokenizers.decoders import WordPiece as DummyDecoder

from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast

from recwizard.utils import DEFAULT_CHAT_TEMPLATE


START_TAG = "<entity>"
END_TAG = "</entity>"


class EntityPreTokenizer:
    def __init__(self):
        import re

        self.pattern = re.compile(rf"{START_TAG}(.*?){END_TAG}")

    def pre_tokenize(self, pretok: PreTokenizedString) -> List[NormalizedString]:
        return pretok.split(self.entity_split)

    def entity_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        """Return the entity tokens within the entity tags"""
        text = str(normalized_string)
        splits = []
        for match in self.pattern.finditer(text):
            start, stop = match.start(0), match.end(0)
            splits.append(normalized_string[start:stop])
        return splits


class EntityBackendTokenizer(BaseTokenizer):

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        pad_token: Union[str, AddedToken] = "[PAD]",
    ):
        if vocab is not None:
            tokenizer = Tokenizer(WordLevel(vocab, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordLevel(unk_token=str(unk_token)))

        parameters = {
            "model": "EntityTokenizer",
            "unk_token": str(unk_token),
            "pad_token": str(pad_token),
        }

        super().__init__(tokenizer, parameters)


class EntityTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        chat_template: str = DEFAULT_CHAT_TEMPLATE,
        *args,
        **kwargs,
    ):

        if vocab is not None:
            if not unk_token in vocab:
                vocab[str(unk_token)] = len(vocab)
            if not pad_token in vocab:
                vocab[str(pad_token)] = len(vocab)
        # Initialize the backend entity tokenizer
        tokenizer = EntityBackendTokenizer(vocab, unk_token=str(unk_token), pad_token=str(pad_token))

        # Initialize the PreTrainedTokenizerFast
        super().__init__(tokenizer_object=tokenizer, *args, **kwargs)

        # Add the custom normalizer, pre-tokenizer and decoder
        self.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(EntityPreTokenizer())

        # Protect the vocab tokens
        self.add_tokens(list(vocab.keys()))

        # Save init_inputs
        self.init_inputs = (vocab,)
        # self.unk_token = unk_token
        # self.pad_token = pad_token
        self.add_special_tokens({"unk_token": unk_token, "pad_token": pad_token})

        # Set the chat template
        if chat_template is not None:
            self.chat_template = chat_template

    def save_pretrained(self, *args, **kwargs) -> Tuple[str]:
        # Save the tokenizer with the dummy pre-tokenizer and decoder
        # To avoid the error: "TypeError: cannot pickle custom objects"

        self.backend_tokenizer.decoder = DummyDecoder()
        self.backend_tokenizer.pre_tokenizer = DummyPreTokenizer()
        return super().save_pretrained(*args, **kwargs)


if __name__ == "__main__":
    import os, json

    from recwizard.utils import create_chat_message

    entity2id = json.load(open("../../../../local_repo/kbrd-rec/raw_vocab/entity2id.json"))

    tokenizer = EntityTokenizer(vocab=entity2id, unk_token="[UNK]", pad_token="[PAD]")

    tokenizer.pad_token = "[PAD]"

    tokenizer.save_pretrained("test")

    tokenizer = EntityTokenizer.from_pretrained("test")

    print(tokenizer.pad_token)

    chat = create_chat_message("User: Hi; <sep> System: Yes! I recommend <entity>Titanic</entity>!")

    print(tokenizer.apply_chat_template(chat, tokenize=False))

    print(tokenizer.apply_chat_template(chat, tokenize=True))
