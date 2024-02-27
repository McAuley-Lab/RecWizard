from typing import List

from nltk import load, NLTKWordTokenizer
from tokenizers import Tokenizer, NormalizedString, PreTokenizedString
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import PreTokenizer
from transformers import PreTrainedTokenizerFast


class NLTKTokenizer:
    def __init__(self, language="english"):
        # nltk.download('punkt')
        self.tokenizer = load(f"tokenizers/punkt/{language}.pickle")
        self.word_tokenizer = NLTKWordTokenizer()

    def word_tokenize(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        return [normalized_string[s:t] for s, t in self.word_tokenizer.span_tokenize(str(normalized_string))]

    def nltk_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        sentences = [normalized_string[s:t] for s, t in self.tokenizer.span_tokenize(str(normalized_string))]
        tokenized = []
        for sentence in sentences:
            tokenized += self.word_tokenize(i, sentence)
        return tokenized

    def pre_tokenize(self, pretok: PreTokenizedString):
        # Let's call split on the PreTokenizedString to split using `self.jieba_split`
        pretok.split(self.nltk_split)


tokenizers = {}


def get_tokenizer(name="redial"):
    return tokenizers[name]


def RnnTokenizer(vocab, name="redial"):
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
    tokenizer = Tokenizer(WordLevel(unk_token='<unk>', vocab=word2id))
    tokenizer.normalizer = Lowercase()

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    wrapped_tokenizer.backend_tokenizer.pre_tokenizer = PreTokenizer.custom(NLTKTokenizer())
    tokenizers[name] = wrapped_tokenizer
    return wrapped_tokenizer


if __name__ == "__main__":
    vocab = ["i", "eat", "apple", "rice"] + ['<pad>', '<s>', '</s>', '<unk>', '\n']
    tokenizer = RnnTokenizer(vocab)
    encoding = tokenizer.encode(["I eat apple today.", "I eat rice today."],
                                padding=True,
                                truncation=True,
                                return_token_type_ids=False,
                                return_tensors='pt',
                                )
    print(encoding)
    from recwizard.tokenizers import NLTKTokenizer
    t2 = NLTKTokenizer(vocab, unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>')
    
    print(t2.encode(["I eat apple today.", "I eat rice today."], padding=True, truncation=True, return_token_type_ids=False,))
    