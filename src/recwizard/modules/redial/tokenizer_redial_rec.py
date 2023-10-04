import re

import torch
from datasets import load_dataset
from transformers import BatchEncoding, AutoTokenizer
from typing import Dict, List, Callable, Union, Tuple

from recwizard.tokenizer_utils import BaseTokenizer, Entity
from recwizard.utility import WrapSingleInput, apply_func, SEP_TOKEN, pad_and_stack


class RedialRecTokenizer(BaseTokenizer):
    date_pattern = re.compile(r'\(\d{4}\)')

    def __init__(self,
                 id2entity: Dict[int, str] = None,
                 sen_encoder="princeton-nlp/unsup-simcse-roberta-base",
                 initiator='User:',
                 respondent='System:',
                 **kwargs
                 ):
        id2entity = {int(k): v for k, v in id2entity.items()}
        super().__init__(tokenizers=[
            AutoTokenizer.from_pretrained(sen_encoder),
        ], id2entity=id2entity)
        self.initiator = initiator
        self.respondent = respondent

    def get_init_kwargs(self):
        return {
            'id2entity': self.id2entity
        }

    @classmethod
    def load_from_dataset(cls, dataset='redial', **kwargs):
        id2entity = loadJsonFileFromDataset(dataset, 'id2entity.json')
        return cls(id2entity=id2entity, **kwargs)

    def preprocess(self, text: str) -> Tuple[str, int]:
        """
        Extract and remove the sender from text
        Args:
            text: an utterance

        Returns: text, sender

        """
        if text.startswith(self.initiator):
            text = text[len(self.initiator):].strip(" ")
            sender = 1
        elif text.startswith(self.respondent):
            text = text[len(self.respondent):].strip(" ")
            sender = -1
        else:
            sender = 0
        return text, sender

    @staticmethod
    def collate_fn(encodings):
        keys = set(encodings[0].keys())
        keys.remove("movie_occurrences")
        res = {
            key: pad_and_stack([e[key] for e in encodings]) for key in keys
        }
        res["movie_occurrences"] = [e["movie_occurrences"] for e in encodings]
        return res

    def encode_plus(
            self,
            text: str,
            *args,
            **kwargs
    ) -> BatchEncoding:
        texts = text.split(SEP_TOKEN)
        # preprocess
        batch_text, senders = apply_func(self.preprocess, texts)
        senders = list(senders)
        # process entity
        processed_results = [self.process_entities(text) for text in batch_text]
        processed_text, batch_entities, movie_names = map(list, zip(*processed_results))
        movieIds = [movieId for entities in batch_entities for movieId in entities]
        movies = [movie for movies in movie_names for movie in movies]
        # replace special token for each tokenzizer
        texts = [list(text) for text in zip(*[self.replace_special_tokens(s) for s in processed_text])]
        # call the encodes function on the list of tokenizer and the list of text
        encodings = self.encodes([tokenizer.batch_encode_plus for tokenizer in self.tokenizers], texts, *args, **kwargs)
        movie_occurrences = torch.stack(
            [self._fill_movie_occurrences(encodings[0], texts[0], movie_name=movie) for movie in movies]) if len(
            movies) > 0 else torch.zeros((0, 0, 1))

        encodings = BatchEncoding({
            'input_ids': encodings[0]['input_ids'],
            'attention_mask': encodings[0]['attention_mask'],
            'senders': torch.as_tensor(senders),
            'movieIds': torch.as_tensor(movieIds),
            'conversation_lengths': torch.as_tensor(len(texts[0])),
            'movie_occurrences': movie_occurrences
        })
        return encodings

    def batch_encode_plus(
            self,
            batch_text_or_text_pairs: List[str],
            *args,
            **kwargs
    ) -> BatchEncoding:
        """
        Args:
            batch_text_or_text_pairs:
            *args:
            **kwargs:

        Returns: BatchEncoding

        """
        encodings = [self.encode_plus(dialog) for dialog in batch_text_or_text_pairs]
        return BatchEncoding(self.collate_fn(encodings))

    def process_entities(self, text: str) -> Tuple[str, List[Entity], List[str]]:
        entities = []
        movie_names = []  # the movie name without year like "(20xx)"
        for m in reversed(list(self.entity_pattern.finditer(text))):  # use reverse to avoid breaking the span
            start, end = m.span()
            entity_name = m.group(1)
            movie_name = self.date_pattern.sub('', entity_name).strip(" ")
            if len(movie_name) > 0:
                text = text[:start] + movie_name + text[end:]
                if self.entity2id is not None and entity_name in self.entity2id:
                    movie_names.append(movie_name)
                    entities.append(self.entity2id[entity_name])
        entities = entities[::-1]
        movie_names = movie_names[::-1]
        return text, entities, movie_names

    def _fill_movie_occurrences(self, encoding, conversation, movie_name):
        max_length = max(len(ex) for ex in encoding["input_ids"])
        movie_occurrences = []
        for i, msg in enumerate(conversation):
            word_ids = encoding[i].word_ids
            occurrence = torch.zeros(max_length)
            # locate the indices of the movie after encoding
            for m in re.finditer(re.escape(movie_name), msg):
                l = word_ids[encoding[i].char_to_token(m.start())]
                r = word_ids[encoding[i].char_to_token(m.end() - 1)]
                occurrence[l: r + 1] = 1
            movie_occurrences.append(occurrence)
        return torch.stack(movie_occurrences)

    def encodes(self, encode_funcs: List[Callable], texts: List[Union[str, List[str]]], *args, **kwargs) -> List[
        BatchEncoding]:
        kwargs.pop('text_pair', None)
        kwargs1 = kwargs.copy()
        kwargs1.update(
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        return [
            encode_funcs[0](texts[0], *args, **kwargs1),
        ]

    @WrapSingleInput
    def decode(
            self,
            ids,
            *args,
            **kwargs,
    ) -> List[str]:
        return [self.id2entity[id] for id in ids]


if __name__ == '__main__':
    dialog = "User: Hello! I'm looking for a horror movie. <sep>System: Hi! Have you watched <entity>Annabelle (2014)</entity>"
    import sys

    sys.path.append('./')
    from recwizard.utility import loadJsonFileFromDataset

    dataset = 'redial'
    datasets = load_dataset(f"dataset/{dataset}", 'formatted')
    tokenizer = RedialRecTokenizer(
        sen_encoder='princeton-nlp/unsup-simcse-roberta-base',
        vocab=loadJsonFileFromDataset(dataset, 'vocab.json'),
        entity2id=loadJsonFileFromDataset(dataset, 'entity2id.json')
    )
    datasets = datasets.map(tokenizer.batch_encode_plus, input_columns='messages')
    print(datasets['test'][0])
