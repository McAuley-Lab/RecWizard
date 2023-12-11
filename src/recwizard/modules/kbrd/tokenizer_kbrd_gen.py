import os
import json
from typing import List, Dict

from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utility.utils import WrapSingleInput

from .tokenizer_nltk import KBRDWordTokenizer
from .tokenizer_kbrd_rec import KBRDRecTokenizer


class KBRDGenTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: List[str],
        id2entity: Dict[int, str] = None,
        **kwargs,
    ):
        """Initialize the KBRDGen tokenizer.
        
        Args:
            vocab (List[str]): list of words in the NLTK tokenizer;
            id2entity (Dict[int, str]): dictionary mapping entity ids to entity names;
        """
        id2entity = {int(k): v for k, v in id2entity.items()}
        super().__init__(
            tokenizers=[
                KBRDWordTokenizer(vocab=vocab),
                KBRDRecTokenizer(id2entity=id2entity),
            ],
            **kwargs,
        )
        self.vocab = vocab
        self.id2entity = {int(k): v for k, v in id2entity.items()}

    def get_init_kwargs(self):
        """
        The kwargs for initialization. Override this function to declare the necessary initialization kwargs (
        they will be saved when the tokenizer is saved or pushed to huggingface model hub.)

        See also: :meth:`~save_vocabulary`
        """
        return {
            "vocab": self.vocab,
            "id2entity": self.id2entity,
        }

    @WrapSingleInput
    def decode(
        self,
        ids,
        *args,
        **kwargs,
    ) -> List[str]:
        """Decode a list of token ids into a list of strings from the NLTK tokenizer."""
        return self.tokenizers[0].decode(ids)

    def __call__(self, *args, **kwargs):
        """Tokenize a string into a list of token ids"""
        kwargs.update(return_tensors="pt", padding=True, truncation=True)
        return super().__call__(*args, **kwargs)