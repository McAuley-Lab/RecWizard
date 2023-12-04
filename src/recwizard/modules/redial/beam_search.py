from collections import Counter

import numpy as np
from nltk import ngrams


def get_best_beam(beams, normalization_alpha=0):
    best_index = 0
    best_score = -1e10
    for (i, beam) in enumerate(beams):
        normalized_score = beam.normalized_score(normalization_alpha)
        if normalized_score > best_score:
            best_index = i
            best_score = normalized_score
    return beams[best_index]


def n_gram_repeats(sequence, n):
    """
    Check if a sequence contains repeated n-grams.

    This function checks if a given sequence contains at least one repeated n-gram.

    Args:
        sequence (iterable): The input sequence to analyze.
        n (int): The size of the n-grams to check for repeats.

    Returns:
        bool: True if the sequence contains at least one repeated n-gram, False otherwise.

    Example:
        >>> n_gram_repeats([1, 2, 3, 4, 1, 2, 3, 4], 2)
        True
    """
    counts = Counter(ngrams(sequence, n))
    if len(counts) == 0:
        return False
    if counts.most_common()[0][1] > 1:
        return True
    return False


class Beam(object):
    end_token = -1

    def __init__(self, sequence, likelihood, mentioned_movies=None):
        if mentioned_movies is None:
            mentioned_movies = set()
        self.finished = False
        self.sequence = sequence
        self.likelihood = likelihood
        self.mentioned_movies = mentioned_movies.copy() if mentioned_movies else set()

    def get_updated_beam(self, token, probability):
        updated_beam = Beam(self.sequence + [token], self.likelihood * probability, self.mentioned_movies)
        if token == self.end_token:
            updated_beam.finished = True
        return updated_beam

    def __str__(self):
        finished_str = "" if self.finished else "not"
        return finished_str + " finished beam of likelihood {} : {}".format(self.likelihood, self.sequence)

    def normalized_score(self, alpha):
        """
        Calculate a normalized score with a length penalty following Wu et al.'s method.

        This method computes a normalized score using a length penalty, as described in the paper:
        "Google's neural machine translation system: Bridging the gap between human and machine translation" by Wu et al.

        Args:
            alpha (float): The alpha parameter controlling the strength of the length penalty.

        Returns:
            float: The normalized score.

        Note:
            - If alpha is 0, the score is not penalized for length.
            - If alpha is greater than 0, a length penalty is applied to the score.
            - The length penalty is based on the length of the sequence in relation to a reference length.

        Reference:
            - Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Dean, J. (2016).
              Google's neural machine translation system: Bridging the gap between human and machine translation.
              arXiv preprint arXiv:1609.08144.
        """
        if alpha == 0:
            return np.log(self.likelihood)
        else:
            penalty = ((5 + len(self.sequence)) / 6) ** alpha
            return np.log(self.likelihood) / penalty


class BeamSearch(object):
    @staticmethod
    def initial_beams(start_sentence):
        """
        Initialize beams for beam search with a start sentence.

        Args:
            start_sentence (list): The initial sequence of tokens for each beam.

        Returns:
            list: A list of Beam instances with the provided start sentences and initial likelihood.

        Example:
            >>> initial_beams(["<bos>"])
            [Beam(sequence=["<bos>"], likelihood=1)]
        """
        return [Beam(sequence=start_sentence, likelihood=1)]

    @staticmethod
    def update_beams(beams, beam_size, probabilities, n_gram_block=None):

        """
        Perform one step of beam search to update the beams.

        Args:
            beams (list): A list of Beam instances representing the current beams.
            beam_size (int): The desired beam size to maintain.
            probabilities (list of tensors): List of beam_size probability tensors (one for each beam).
            n_gram_block (int, optional): The n-gram block size for preventing n-gram repeats.

        Returns:
            list: A list of the new beams after the update step.

        Note:
            The function selects the top-k beams based on their likelihood and updates them.
        """

        vocab_size = probabilities[0].data.shape[0]
        # compute the likelihoods for the next token
        # vector for finished beams. First dimension will be the likelihood of the finished beam, other dimensions are
        # zeros so this beam is counted only once in the top k
        finsished_beam_vec = np.zeros(vocab_size)
        finsished_beam_vec[0] = 1
        # (beam_size, vocab_size)
        new_probabilities = np.array([beam.likelihood * probability.data.numpy() if not beam.finished
                                      else beam.likelihood * finsished_beam_vec
                                      for beam, probability in zip(beams, probabilities)])
        # get the top-k (beam_size) probabilities
        ind = np.unravel_index(np.argsort(new_probabilities, axis=None), new_probabilities.shape)
        # inspect hypothesis in descending order of likelihood
        ind = (ind[0][::-1], ind[1][::-1])
        # get the list of top-k updated beams
        new_beams = []
        for beam_index, token in zip(*ind):
            # if finished, append the beam as is
            if beams[beam_index].finished:
                new_beams.append(beams[beam_index])
            # otherwise, update the beam with the chosen token
            else:
                # check n_gram blocking. Note that n_gram blocking is not used to produce the results in the article
                if n_gram_block is None or not n_gram_repeats(beams[beam_index].sequence + [token], n_gram_block):
                    # add extended hypothesis to new_beam list
                    new_beams.append(
                        beams[beam_index].get_updated_beam(token, probabilities[beam_index][token].data.numpy()))

            # return when beam_size valid beams found
            if len(new_beams) >= beam_size:
                return new_beams
        return new_beams
