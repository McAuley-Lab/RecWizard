import os
import json
from typing import List

from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utility.utils import WrapSingleInput

from .tokenizer_nltk import KBRDWordTokenizer
from .tokenizer_kbrd_rec import KBRDRecTokenizer


class KBRDGenTokenizer(BaseTokenizer):
    init_word_list_file = "nltk_word_list.json"

    def __init__(
        self,
        word_tokenizer: BaseTokenizer = None,
        entity_tokenizer: BaseTokenizer = None,
        **kwargs,
    ):
        self.word_tokenizer = word_tokenizer
        self.entity_tokenizer = entity_tokenizer
        tokenizers = [self.word_tokenizer, self.entity_tokenizer]

        if word_tokenizer is None or entity_tokenizer is None:
            raise ValueError("word_tokenizer and entity_tokenizer should not be None.")

        super().__init__(tokenizers=tokenizers, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                pretrained_model_name_or_path, cls.init_word_list_file
            )
        except:
            path = os.path.join(pretrained_model_name_or_path, cls.init_word_list_file)
        word_tokenizer = KBRDWordTokenizer(json.load(open(path, "r")))
        entity_tokenizer = KBRDRecTokenizer.from_pretrained(
            pretrained_model_name_or_path
        )
        return cls(
            word_tokenizer=word_tokenizer,
            entity_tokenizer=entity_tokenizer,
            *args,
            **kwargs,
        )

    @WrapSingleInput
    def decode(
        self,
        ids,
        *args,
        **kwargs,
    ) -> List[str]:
        return self.tokenizers[0].decode(ids)

    def __call__(self, *args, **kwargs):
        kwargs.update(return_tensors="pt", padding=True, truncation=True)
        return super().__call__(*args, **kwargs)
