from typing import List

import nltk
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import PreTrainedTokenizerFast


class NLTKTokenizer:
    def __init__(self, language="english"):
        """Initialize the NLTK tokenizer.

        Args:
            language(str): language to use for the tokenizer
        """
        # nltk.download('punkt')
        st_path = f"tokenizers/punkt/{language}.pickle"
        try:
            self.tokenizer = nltk.data.load(st_path)
        except LookupError:
            nltk.download("punkt")
            self.tokenizer = nltk.data.load(st_path)
        self.word_tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()

    def word_tokenize(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        """Tokenize a string into words."""
        return [
            normalized_string[s:t]
            for s, t in self.word_tokenizer.span_tokenize(str(normalized_string))
        ]

    def nltk_split(
        self, i: int, normalized_string: NormalizedString
    ) -> List[NormalizedString]:
        """Split a string into sentences using NLTK."""
        sentences = [
            normalized_string[s:t]
            for s, t in self.tokenizer.span_tokenize(str(normalized_string))
        ]
        tokenized = []
        for sentence in sentences:
            tokenized += self.word_tokenize(i, sentence)
        return tokenized

    def pre_tokenize(self, pretok: PreTokenizedString):
        """Pre-tokenize a string into sentences using NLTK."""
        # Let's call split on the PreTokenizedString to split using `self.jieba_split`
        pretok.split(self.nltk_split)


tokenizers = {}


def get_tokenizer(name="kbrd"):
    """Return a tokenizer from the cache."""
    return tokenizers[name]


def KBRDWordTokenizer(vocab, name="kbrd"):
    """
    Return a tokenizer for RNN models from the given vocabulary
    Args:
        vocab(List[str]): list of words
        name(str): name of the tokenizer. Used to cache the tokenizer

    Returns:
        PreTrainedTokenizerFast
    """
    if tokenizers.get(name):
        return tokenizers[name]
    word2id = {word: i for i, word in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(unk_token="__unk__", vocab=word2id))
    tokenizer.normalizer = Lowercase()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="__unk__",
        pad_token="__null__",
        bos_token="__start__",
        eos_token="__end__",
    )
    wrapped_tokenizer.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(
        NLTKTokenizer()
    )
    tokenizers[name] = wrapped_tokenizer
    return wrapped_tokenizer
