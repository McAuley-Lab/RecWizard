import csv, re
import os
import pickle
from collections import Counter

import h5py
import nltk
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from recwizard.utility import pad_and_stack, loadJsonFileFromDataset
from recwizard.modules.redial.tokenizer_rnn import RnnTokenizer

def tokenize(message):
    """
    Text processing: Sentence tokenize, then concatenate the word_tokenize of each sentence. Then lower.
    :param message:
    :return:
    """
    sentences = nltk.sent_tokenize(message)
    tokenized = []
    for sentence in sentences:
        tokenized += nltk.word_tokenize(sentence)
    return [word.lower() for word in tokenized]

class RedialDataProcessor:
    id2name = {}
    db2id = {}
    mlId2index = {}
    id_pattern = re.compile(r'@(\d+)')
    _has_movie_mappings = False

    def __init__(self, tokenizer=None, tokenize_for_rnn=False, dataset='redial'):
        """
        :param tokenizer: the transformer tokenizer
        :param tokenize_for_rnn: bool. If True, will place word ids in example["dialgoue"] for rnn.
        :param cache_dir: the directory for saving temporary cache
        """
        super().__init__()
        self.tokenizer = tokenizer
        self._load_movie_mappings()
        # build vocabulary for RNN
        self.tokenize_for_rnn = tokenize_for_rnn
        self.utterance_length_limit = 80
        self.conversation_length_limit = 40
        if self.tokenize_for_rnn:
            self.vocab = loadJsonFileFromDataset(dataset, 'vocab.json')
            self.rnn_tokenizer = RnnTokenizer(self.vocab)
            # self.vocab_size = len(self.vocab)
            # self.word2id = {word: i for i, word in enumerate(self.vocab)}
            # self.id2word = {i: word for word, i in self.word2id.items()}

    def _load_movie_mappings(self, path="dataset/redial/movies_merged.csv"):
        if self._has_movie_mappings:
            return
        with open(path, 'r') as f:
            reader = csv.reader(f)
            # remove date from movie name
            date_pattern = re.compile(r'\(\d{4}\)')
            for row in reader:
                if row[0] != "index":
                    self.id2name[int(row[0])] = date_pattern.sub('', row[1]).strip(" ")
                    # id2name[int(row[0])] = row[1]
                    self.db2id[int(row[2])] = int(row[0])
                    self.mlId2index[int(row[3])] = int(row[0])
        del self.db2id[-1]
        print("loaded {} movies from {}".format(len(self.id2name), path))
        # get dataset characteristics
        self.db2name = {db: self.id2name[id] for db, id in self.db2id.items()}
        self.n_redial_movies = len(self.db2id.values())  # number of movies mentioned in ReDial
        self.n_ml_movies = len(self.id2name)
        self._has_movie_mappings = True

    def replace_movie_with_name(self, txt):
        return self.id_pattern.sub(lambda m: self.db2name[int(m.group(1))], txt)

    def replace_movie_with_words(self, tokens):
        """
        If the ID corresponds to a movie, returns the sequence of tokens that correspond to this movie name
        :param tokens:
        :return: modified sequence
        """
        res = []
        for token_id in tokens:
            if token_id <= len(self.vocab):
                res.append(token_id)
            else:
                res.extend(self.rnn_tokenize(self.id2name[token_id - len(self.vocab)]))
        return res
    def transformer_tokenize(self, sentences):
        return self.tokenizer(sentences,
                              padding=True,
                              truncation=True,
                              return_token_type_ids=False,
                              return_tensors='pt',
                              )
    def rnn_tokenize(self, sentences):
        return self.rnn_tokenizer(sentences,
                              padding=True,
                              truncation=True,
                              return_token_type_ids=False,
                              add_special_tokens=False,
                              )

    def token2id(self, token):
        """
        :param token: string or movieId
        :return: corresponding ID
        """

        if isinstance(token, int):
            return token
        return self.rnn_tokenizer.convert_tokens_to_ids(token)


    def _fill_movie_occurrences(self, encoding, conversation, movie_name):
        max_length = max(len(ex) for ex in encoding["input_ids"])
        movie_occurrences = []
        for i, msg in enumerate(conversation):
            word_ids = encoding[i].word_ids
            occurrence = torch.zeros(max_length)
            # locate the indices of the movie after encoding
            for m in re.finditer(re.escape(movie_name), msg):
                l = word_ids[encoding[i].char_to_token(m.start())]
                r = word_ids[encoding[i].char_to_token(m.end()-1)]
                occurrence[l: r+1] = 1
            movie_occurrences.append(occurrence)
        return torch.stack(movie_occurrences)
    def _replace_movies_in_tokenized(self, tokenized):
        """
        replace movieId tokens in a single tokenized message.
        Eventually compute the movie occurrences and the target with (global) movieIds
        :param tokenized:
        :return:
        """
        output_with_id = tokenized[:]
        pattern = re.compile(r'^\d{5,6}$')
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            # Check if word corresponds to a movieId.
            if pattern.match(word) and int(word) in self.db2id:
                # get the global Id
                movieId = self.db2id[int(word)]
                # remove ID
                del tokenized[index]
                # put tokenized movie name instead. len(tokenized_movie) - 1 elements are added to tokenized.
                tokenized_movie = tokenize(self.id2name[movieId])
                tokenized[index:index] = tokenized_movie

                # update output_with_id: replace word with movieId repeated as many times as there are words in the
                # movie name. Add the size-of-vocabulary offset.
                output_with_id[index:index + 1] = [movieId + len(self.vocab)] * len(tokenized_movie)

                # increment index
                index += len(tokenized_movie)

            else:
                # do nothing, and go to next word
                index += 1

        return tokenized, output_with_id


    def map_redial_for_recommender(self, example):
        example["messages"] = example["messages"][:self.conversation_length_limit]
        example["senders"] = example["senders"][:self.conversation_length_limit]
        texts = [self.replace_movie_with_name(msg) for msg in example["messages"]]

        # tokenize for transformer
        encoding = self.transformer_tokenize(texts)

        # fill movie_occurences after tokenization
        movieIds = [self.db2id[dbId] for dbId in example["movieIds"] if len(self.db2name.get(dbId, "")) > 0]
        movie_occurrences = [self._fill_movie_occurrences(encoding, texts, movie_name=self.id2name[id]) for id in movieIds]
        dialogue = []
        target = []
        raw_texts = []

        for txt in example["messages"]:
            pattern = re.compile(r'@(\d+)')
            message_text = pattern.sub(lambda m: " " + m.group(1) + " ", txt)  # e.g. replace @2019 with 2019
            text = tokenize(message_text)
            text = text[:self.utterance_length_limit]
            text, message_target = self._replace_movies_in_tokenized(text)
            dialogue.append(['<s>'] + text + ['</s>'])
            target.append(message_target + ['</s>', '</s>']) # shift the target by 1
            raw_texts.append(text)

        lengths = [len(s) for s in dialogue]
        # tokenize for rnn
        dialogue = pad_and_stack([torch.tensor([self.token2id(token) for token in s]) for s in dialogue])
        target = pad_and_stack([torch.tensor([self.token2id(token) for token in s]) for s in target])

        return {
            "raw_texts": raw_texts,
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "movieIds": movieIds,
            "movie_occurrences": movie_occurrences,
            "senders": example["senders"],
            "target": target,
            "dialogue": dialogue,
            "lengths": lengths,
            "conversation_lengths": len(lengths)
        }



    def map_redial_for_sentiment_analysis(self, example):
        texts = [self.replace_movie_with_name(msg) for msg in example["messages"]]

        # tokenize for transformer
        encoding = self.transformer_tokenize(texts)
        # apply nltk tokenize
        dialogue = [tokenize(s) for s in texts]

        # fill movie_occurences after tokenization
        movie_occurrences = self._fill_movie_occurrences(encoding, texts, movie_name=self.db2name[example["movieId"]])

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "movie_occurrences": movie_occurrences,
            "senders": example["senders"],
            "labels": example["form"],
            "lengths": torch.tensor([len(s) for s in dialogue]),
            "conversation_lengths": len(dialogue)
        }

    @staticmethod
    def load_movies_merged(path="dataset/redial/movies_merged.csv"):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            id2index = {int(row[3]): int(row[0]) for row in reader if row[0] != "index"}
        return id2index

    @staticmethod
    def process_rating(rating, binary_rating):
        if binary_rating:
            # return 1 for ratings >= 2.5, 0 for lower ratings (this gives 87% of liked on movielens-latest)
            # return 1 for ratings >= 2, 0 for lower ratings (this gives 94% of liked on movielens-latest)
            return float(float(rating) >= 2)
        # return a rating between 0 and 1
        return (float(rating) - .5) / 4.5


    def get_task_embedding(self, pretrained_emb='embeddings/glove.840B.300d.h5'):
        pretrained_embeddings = h5py.File(pretrained_emb)
        embedding_matrix = pretrained_embeddings['embedding'][()]
        pretrain_vocab = pretrained_embeddings['words_flatten'][()].decode().split('\n')
        pretrain_word2id = {
            word: ind for ind, word in enumerate(pretrain_vocab)
        }
        task_embeddings = []
        oov = 0
        for word in self.vocab:
            if word in pretrain_word2id:
                task_embeddings.append(
                    embedding_matrix[pretrain_word2id[word]]
                )
            else:
                oov += 1
                task_embeddings.append(
                    np.zeros(300, dtype=np.float32)
                )
        print(f"{len(self.vocab)} words, {oov} oov")
        return np.stack(task_embeddings).astype(np.float32)

