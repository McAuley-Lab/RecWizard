from typing import Dict, List, Callable, Union
from transformers import AutoTokenizer, BatchEncoding

from recwizard.utility.utils import WrapSingleInput, loadJsonFileFromDataset
from recwizard.tokenizer_utils import BaseTokenizer

gpt2_special_tokens_dict = {
    'pad_token': '<pad>',
    'additional_special_tokens': ['<movie>'],
}

prompt_special_tokens_dict = {
    'additional_special_tokens': ['<movie>'],
}


class UnicrsRecTokenizer(BaseTokenizer):

    def __init__(
            self,
            context_tokenizer: str = "microsoft/DialoGPT-small",
            prompt_tokenizer: str = "roberta-base",
            context_max_length: int = 200,
            prompt_max_length: int = 200,
            pad_entity_id: int = 31161,
            entity2id: Dict[str, int] = None,
            **kwargs,
    ):
        context_tokenizer = AutoTokenizer.from_pretrained(context_tokenizer, truncation_side='left')
        prompt_tokenizer = AutoTokenizer.from_pretrained(prompt_tokenizer, truncation_side='left')
        context_tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        tokenizers = [
            context_tokenizer,
            prompt_tokenizer
        ]
        super().__init__(tokenizers=tokenizers, entity2id=entity2id, pad_entity_id=pad_entity_id, **kwargs)
        self.context_max_length = context_max_length
        self.prompt_max_length = prompt_max_length

    def __call__(self, *args, **kwargs):
        kwargs.update(return_tensors='pt', padding=True, truncation=True)
        return super().__call__(*args, **kwargs)

    @classmethod
    def load_from_dataset(cls, dataset='redial_unicrs', **kwargs):
        entityName2id = loadJsonFileFromDataset(dataset, 'entityName2id.json')
        entity2id = loadJsonFileFromDataset(dataset, 'entity2id.json')
        return cls(entity2id=entityName2id, pad_entity_id=max(entity2id.values()) + 1, **kwargs)

    @staticmethod
    def mergeEncoding(encodings: List[BatchEncoding]) -> BatchEncoding:
        res = encodings[0].copy()
        if len(encodings) == 0:
            return res
        res.data = {
            'context': encodings[0],
            'prompt': encodings[1]
        }
        return res

    def encodes(self, encode_funcs: List[Callable], texts: List[Union[str, List[str]]], *args, **kwargs) -> List[
        BatchEncoding]:
        kwargs1 = kwargs.copy()
        kwargs1.update(
            max_length=self.context_max_length
        )
        kwargs2 = kwargs.copy()
        kwargs2.update(
            max_length=self.prompt_max_length
        )
        return [
            encode_funcs[0](texts[0], *args, **kwargs1),
            encode_funcs[1](texts[1], *args, **kwargs2),
        ]

    @WrapSingleInput
    def decode(
            self,
            ids,
            *args,
            **kwargs,
    ) -> List[str]:
        return [self.id2entity[id] for id in ids]
