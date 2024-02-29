from typing import Tuple
import torch
import re

initiator = "User:"
respondent = "System:"


def preprocess(text: str) -> Tuple[str, int]:
    """
    Extract and remove the sender from text
    Args:
        text: an utterance

    Returns: text, sender

    """
    if text.startswith(initiator):
        text = text[len(initiator) :].strip(" ")
        sender = 1
    elif text.startswith(respondent):
        text = text[len(respondent) :].strip(" ")
        sender = -1
    else:
        sender = 0
    return text, sender


def fill_movie_occurrences(encoding, conversation, movie_name):
    max_length = max(len(ex) for ex in encoding["input_ids"])
    movie_occurrences = []
    for i, msg in enumerate(conversation):
        word_ids = encoding[i].word_ids
        occurrence = torch.zeros(max_length)
        # locate the indices of the movie after encoding
        for m in re.finditer(re.escape(movie_name), msg):
            l = word_ids[encoding[i].char_to_token(m.start())]
            r = word_ids[encoding[i].char_to_token(m.end() - 1)]
            occurrence[l : r + 1] = 1
        movie_occurrences.append(occurrence)
    return torch.stack(movie_occurrences)


def pad_and_stack(tensors, dtype=torch.long, pad_value=0):
    """
    Pad tensors of different sizes with zeros and stack them up.

    Args:

        tensors (list of torch.Tensor): List of tensors to be padded and stacked.
            Each tensor should have the same number of dimensions, but different
            sizes in the last dimension.
        dtype (torch.dtype, optional): Data type of the output tensor.
        pad_value (int, optional): The value to fill in the padded areas.

    Returns:
        padded_and_stacked (torch.Tensor): The padded and stacked tensor, with
            shape (num_tensors, max_size_dim1, max_size_dim2, ..., max_size_dimN),
            where N is the number of dimensions of the input tensors.
    """

    # Get the maximum size of each dimension across all tensors
    device = tensors[0].device
    max_sizes = [max(tensor.size(d) for tensor in tensors) for d in range(len(tensors[0].size()))]

    # Create a tensor filled with zeros, with size (num_tensors, max_size_dim1, max_size_dim2, ..., max_size_dimN)
    padded_and_stacked = pad_value * torch.ones((len(tensors), *max_sizes), dtype=dtype, device=device)

    # Pad each tensor with zeros and add it to the padded_and_stacked tensor
    for i, tensor in enumerate(tensors):
        slices = [slice(0, size) for size in tensor.size()]
        padded_and_stacked[(i,) + tuple(slices)] = tensor
    return padded_and_stacked


def sort_for_packed_sequence(lengths):
    """
    Sorts an array of lengths in descending order and returns the sorted lengths,
    the indices to sort, and the indices to retrieve the original order.

    Args:
        lengths: 1D array of lengths

    Returns:
        sorted_lengths: lengths in descending order
        sorted_idx: indices to sort
        rev: indices to retrieve original order
    """
    sorted_idx = torch.argsort(lengths, descending=True)  # idx to sort by length
    sorted_lengths = lengths[sorted_idx]
    rev = torch.argsort(sorted_idx)  # idx to retrieve original order

    return sorted_lengths, sorted_idx, rev


def get_task_embedding(vocab, pretrained_emb="embeddings/glove.840B.300d.h5"):
    import h5py
    import numpy as np

    pretrained_embeddings = h5py.File(pretrained_emb)
    embedding_matrix = pretrained_embeddings["embedding"][()]
    pretrain_vocab = pretrained_embeddings["words_flatten"][()].decode().split("\n")
    pretrain_word2id = {word: ind for ind, word in enumerate(pretrain_vocab)}
    task_embeddings = []
    oov = 0
    for word in vocab:
        if word in pretrain_word2id:
            task_embeddings.append(embedding_matrix[pretrain_word2id[word]])
        else:
            oov += 1
            task_embeddings.append(np.zeros(300, dtype=np.float32))
    logging.info(f"{len(vocab)} words, {oov} oov")
    return np.stack(task_embeddings).astype(np.float32)
