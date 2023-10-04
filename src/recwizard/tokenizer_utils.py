import json
import os
import re
from typing import Optional, Tuple, Dict, List, Union, Callable

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from huggingface_hub import hf_hub_download

from recwizard.utility import SEP_TOKEN, BOS_TOKEN, EOS_TOKEN, ENTITY_PATTERN, loadJsonFileFromDataset, pad_and_stack

Entity = int


class BaseTokenizer(PreTrainedTokenizer):
    init_kwargs_file = 'tokenizer_kwargs.json'
    pattern_sep_token = re.compile(SEP_TOKEN)
    pattern_bos_token = re.compile(BOS_TOKEN)
    pattern_eos_token = re.compile(EOS_TOKEN)

    def __init__(self,
                 entity2id: Dict[str, int] = None,
                 id2entity: Dict[int, str] = None,
                 pad_entity_id: int = None,
                 tokenizers: Union[List[PreTrainedTokenizerBase], PreTrainedTokenizerBase] = None,
                 **kwargs):
        """

        Args:
            entity2id (Dict[str, int]): a dict mapping entity name to entity id. If not provided, it will be generated from id2entity.
            id2entity (Dict[int, str]): a dict mapping entity id to entity name. If not provided, it will be generated from entity2id.
            pad_entity_id (int): the id for padding entity. If not provided, it will be the maximum entity id + 1.
            tokenizers (List[PreTrainedTokenizerBase]): a list of tokenizers to be used.
            **kwargs: other arguments for PreTrainedTokenizer
        """
        super().__init__(
            **kwargs
        )
        if entity2id is None and id2entity is not None:
            entity2id = {v: k for k, v in id2entity.items()}
        if id2entity is None and entity2id is not None:
            id2entity = {v: k for k, v in entity2id.items()}
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.pad_entity_id = pad_entity_id if pad_entity_id is not None else max(self.entity2id.values()) + 1
        if tokenizers is None:
            self.tokenizers = None
        else:
            self.tokenizers = tokenizers if isinstance(tokenizers, List) else [tokenizers]
        self.entity_pattern = re.compile(ENTITY_PATTERN)

    @classmethod
    def load_from_dataset(cls, dataset='redial_unicrs', **kwargs):
        """
        Initialize the tokenizer from the dataset. By default, it will load the entity2id from the dataset.
        Args:
            dataset: the dataset name
            **kwargs: the other arguments for initialization

        Returns:
            (BaseTokenizer): the initialized tokenizer
        """
        entity2id = loadJsonFileFromDataset(dataset, 'entity2id.json')
        return cls(entity2id=entity2id, **kwargs)

    def unk_token(self) -> str:
        """
        Override this function if you want to change the unk_token.
        """
        return "<unk>"

    def unk_token_id(self) -> Optional[int]:
        """
        Override this function if you want to change the unk_token.
        """
        return -1

    @property
    def vocab_size(self) -> int:
        return len(self.entity2id)

    def _convert_token_to_id(self, token: str) -> int:
        return self.entity2id[token] if token in self.entity2id else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2entity[index] if index in self.id2entity else self.unk_token

    @staticmethod
    def mergeEncoding(encodings: List[BatchEncoding]) -> BatchEncoding:
        """
        Merge a list of encodings into one encoding.
        Assumes each encoding has the same attributes other than data.
        """
        if len(encodings) == 0:
            return BatchEncoding({})
        res = encodings[0].copy()
        for encoding in encodings[1:]:
            res.data.update(encoding.data)
        return res

    def replace_special_tokens(self, text: str) -> List[str]:
        """
        Replace the cls token, sep token and eos token for each tokenizer

        Args:
            text: the text to be replaced

        Returns:
            (List[str]): a list of text, each used for one tokenizer
        """

        def replace_for_tokenizer(text: str, tokenizer: PreTrainedTokenizerBase):
            text = self.replace_bos_token(text, tokenizer)
            text = self.replace_sep_token(text, tokenizer)
            text = self.replace_eos_token(text, tokenizer)
            return text

        return [replace_for_tokenizer(text, tokenizer) for tokenizer in self.tokenizers]

    def replace_sep_token(self, text, tokenizer: PreTrainedTokenizerBase):
        token = str(tokenizer._sep_token or tokenizer.eos_token)
        text = self.pattern_sep_token.sub(token, text)
        return text

    def replace_bos_token(self, text, tokenizer: PreTrainedTokenizerBase):
        token = str(tokenizer._cls_token or '')
        text = self.pattern_bos_token.sub(token, text)
        return text

    def replace_eos_token(self, text, tokenizer: PreTrainedTokenizerBase):
        token = str(tokenizer._eos_token or ' ')
        text = self.pattern_eos_token.sub(token, text)
        return text

    def encodes(self, encode_funcs: List[Callable], texts: List[Union[str, List[str]]], *args, **kwargs) -> List[
        BatchEncoding]:
        """
        This function is called to apply encoding functions from different tokenizers. It will be
        used by both `encode_plus` and `batch_encode_plus`.

        If you want to call different tokenizers with different arguments, override this method.

        Args:
            encode_funcs: the encoding functions from `self.tokenizers`.
            texts: the processed text for each encoding function
            **kwargs:

        Returns:
            a list of BatchEncoding, the length of the list is the same as the number of tokenizer
        """
        assert len(encode_funcs) == len(texts)
        return [func(text, *args, **kwargs) for text, func in zip(texts, encode_funcs)]

    def preprocess(self, text: str) -> str:
        """
        Override this function to preprocess the text. It will be used by both `encode_plus` and `batch_encode_plus`.

        Args:
            text: the text to be preprocessed
        Returns: processed text

        """
        return text

    def batch_encode_plus(
            self,
            batch_text_or_text_pairs: List[str],
            *args,
            **kwargs
    ) -> BatchEncoding:
        """
        Overrides the `batch_encode_plus` function from `PreTrainedTokenizer` to support entity processing.
        """
        # preprocess
        batch_text = map(self.preprocess, batch_text_or_text_pairs)
        # process entity
        processed_results = [self.process_entities(text) for text in batch_text]
        processed_text, batch_entities = map(list, zip(*processed_results))
        if kwargs.get('padding') == True:
            batch_entities = pad_and_stack([torch.tensor(entities, dtype=torch.long) for entities in batch_entities],
                                           pad_value=self.pad_entity_id)
            if kwargs.get('return_tensors') is None:
                batch_entities = batch_entities.tolist()
        if self.tokenizers is None:
            return BatchEncoding({
                'raw_text': processed_text,
                'entities': batch_entities
            })
        # replace special tokens for each tokenzizer
        texts = [list(text) for text in zip(*[self.replace_special_tokens(s) for s in processed_text])]
        # call the encodes function on the list of tokenizer and the list of text
        encodings = self.encodes([tokenizer.batch_encode_plus for tokenizer in self.tokenizers], texts, *args, **kwargs)
        encodings = self.mergeEncoding(encodings)
        # add entities to the encodings
        encodings.data['entities'] = batch_entities
        return encodings

    def encode_plus(
            self,
            text: str,
            *args,
            **kwargs
    ) -> BatchEncoding:
        """
        Overrides the `encode_plus` function from `PreTrainedTokenizer` to support entity processing.
        """
        text = self.preprocess(text)
        processed_text, entities = self.process_entities(text)
        if self.tokenizers is None:
            return BatchEncoding({
                'raw_text': processed_text,
                'entities': entities
            })
        texts = self.replace_special_tokens(processed_text)
        encodings = self.encodes([tokenizer.encode_plus for tokenizer in self.tokenizers], texts, *args, **kwargs)
        encodings = self.mergeEncoding(encodings)
        if kwargs.get('return_tensors') == 'pt':
            entities = torch.tensor(entities, dtype=torch.long)
        encodings.data['entities'] = entities
        return encodings

    def process_entities(self, text: str) -> Tuple[str, List[Entity]]:
        """
        Process the entities in the text. It extracts the entity ids from the text and remove the entity tags.
        """
        entities = []
        for m in reversed(list(self.entity_pattern.finditer(text))):  # use reverse to avoid breaking the span
            start, end = m.span()
            entity_name = m.group(1).strip(" ")
            text = text[:start] + entity_name + text[end:]
            if self.entity2id is not None and entity_name in self.entity2id:
                entities.append(self.entity2id[entity_name])
        entities = entities[::-1]
        return text, entities

    def decode(
            self,
            *args,
            **kwargs,
    ) -> str:
        """
        Overrides the `decode` function from `PreTrainedTokenizer`. By default, calls the `decode` function of the first tokenizer.
        """
        return self.tokenizers[0].decode(*args, **kwargs)

    def get_init_kwargs(self):
        """
        The kwargs for initialization. Override this function to declare the necessary initialization kwargs (
        they will be saved when the tokenizer is saved or pushed to huggingface model hub.)

        See also: :meth:`~save_vocabulary`
        """
        return {
            'entity2id': self.entity2id,
            'pad_entity_id': self.pad_entity_id
        }

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        This method is overridden to save the initialization kwargs to the model directory.
        """
        filename_prefix = filename_prefix + "-" if filename_prefix else ""
        vocab_path = os.path.join(save_directory, filename_prefix, self.init_kwargs_file)
        json.dump(self.get_init_kwargs(), open(vocab_path, 'w'))
        return str(vocab_path),

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the tokenizer from pretrained model or local directory.
        It loads the initialization kwargs from the 'tokenizer_kwargs.json' file before initializing the tokenizer.
        """
        try:
            path = hf_hub_download(pretrained_model_name_or_path, cls.init_kwargs_file)
        except:
            path = os.path.join(pretrained_model_name_or_path, cls.init_kwargs_file)
        init_kwargs = json.load(open(path, 'r'))
        kwargs.update(init_kwargs)
        return cls(*args, **kwargs)


if __name__ == '__main__':
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BaseTokenizer(entity2id={'movie': 0}, tokenizers=bert_tokenizer)
    s = ["The <item>movie</item> is interesting<sep>Yes! I think so"]
    print(tokenizer(s[0], return_token_type_ids=False))
    save_dir = "dummy_tokenizer"
    tokenizer.save_pretrained(save_dir)
    tokenizer = BaseTokenizer.from_pretrained(save_dir, tokenizers=bert_tokenizer)
    print(tokenizer(s, return_token_type_ids=False))
