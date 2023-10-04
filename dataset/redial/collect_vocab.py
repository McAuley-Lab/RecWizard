import os, sys
import nltk
from collections import Counter
import pickle
from datasets import load_dataset
from tqdm import tqdm
import csv
import json
import re

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


def load_movie_mappings(path):
    id2name = {}
    db2id = {}

    with open(path, 'r') as f:
        reader = csv.reader(f)
        # remove date from movie name
        for row in reader:
            if row[0] != "index":
                id2name[int(row[0])] = row[1]
                # id2name[int(row[0])] = row[1]
                db2id[int(row[2])] = int(row[0])

    del db2id[-1]
    date_pattern = re.compile(r'\(\d{4}\)')

    # get dataset characteristics
    db2name = {db: date_pattern.sub('', id2name[id]).strip(" ") for db, id in db2id.items()}
    n_redial_movies = len(db2id.values())  # number of movies mentioned in ReDial
    # name2id = {name: int(i) for i, name in id2name.items() if name != ''}

    # print("loaded {} movies from {}".format(len(name2id), path))
    return id2name, db2name


def get_vocab(dataset, db2name):
    """
    get the vocabulary from the train data
    :return: vocabulary
    """
    print(f"Loading vocabulary from {dataset} dataset")
    counter = Counter()
    # get vocabulary from dialogues
    datasets = load_dataset(dataset, download_mode="force_redownload")
    date_pattern = re.compile(r'@(\d+)')
    for subset in ["train", "validation", "test"]:
        for conversation in tqdm(datasets[subset]):
            for message in conversation["messages"]:
                # remove movie Ids
                text = tokenize(date_pattern.sub(" ", message))
                counter.update([word.lower() for word in text])
    # get vocabulary from movie names
    for movieId in db2name:
        tokenized_movie = tokenize(db2name[movieId])
        counter.update([word.lower() for word in tokenized_movie])
    # Keep the most common words
    kept_vocab = counter.most_common(15000)
    vocab = [x[0] for x in kept_vocab]
    print("Vocab covers {} word instances over {}".format(
        sum([x[1] for x in kept_vocab]),
        sum([counter[x] for x in counter])
    ))
    # note: let the <pad> token corresponds to 0
    vocab = ['<pad>', '<s>', '</s>', '<unk>', '\n'] + vocab

    return vocab

if __name__ == '__main__':
    import os
    dataset = 'redial'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    id2entity, db2name = load_movie_mappings(os.path.join(base_dir, "movies_merged.csv"))

    with open(os.path.join(base_dir, 'id2entity.json'), 'w') as f:
        json.dump(id2entity, f)
    # vocab = get_vocab(dataset, db2name)
    # print("vocab has length:", len(vocab))
    # with open(os.path.join(base_dir, 'vocab.json'), 'w') as f:
    #     json.dump(vocab, f)
    #