import re

import torch
from torch import nn as nn
from torch.nn import functional as F
from recwizard.model_utils import BasePipeline
from .configuration_switch_decode import SwitchDecodeConfig
from recwizard.modules.redial import Beam, BeamSearch, get_best_beam
from ...utility import SEP_TOKEN


class SwitchDecodePipeline(BasePipeline):
    """
    A pipeline for response generation using a switching mechanism to integrate movie recommendations.
    """
    config_class = SwitchDecodeConfig

    def __init__(self, config, temperature=1, sample_movies=False, **kwargs):
        """
        Initialize the SwitchDecodePipeline.

        Args:
            config (SwitchDecodeConfig): An instance of SwitchDecodeConfig containing pipeline configuration.
            temperature (float, optional): Temperature parameter for controlling randomness in generation. Defaults to 1.
            sample_movies (bool, optional): Whether to sample movies during generation. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the BasePipeline constructor.
        """
        super().__init__(config, **kwargs)
        self.switch = nn.Linear(in_features=self.config.hidden_size + self.config.context_size, out_features=1)
        self.vocab_size = len(self.gen_tokenizer)
        self.context_size = self.gen_module.decoder.context_size
        self.sample_movies = sample_movies
        self.t = temperature
        self.context = None
        self.recs = None


    def switch_decode(self, switch_context, language_output, movie_recommendations, temperature, forbid_movies=None):
        """
        Perform switch decoding to integrate movie recommendations into the generated response.

        Args:
            switch_context (torch.Tensor): Switch context tensor.
            language_output (torch.Tensor): Language model output tensor.
            movie_recommendations (torch.Tensor): Movie recommendation tensor.
            temperature (float): Temperature parameter for controlling randomness.
            forbid_movies (set, optional): Set of forbidden movie indices. Defaults to None.

        Returns:
            torch.Tensor: Switched output tensor.
        """
        batch_size, max_utterance_length = language_output.shape[:2]
        n_movies = movie_recommendations.data.shape[1]

        max_probabilities, _ = torch.max(F.softmax(movie_recommendations, dim=1), dim=1)

        movie_recommendations = movie_recommendations.unsqueeze(1).expand(
            batch_size, max_utterance_length, n_movies).contiguous()
        switch = self.switch(switch_context)
        # For generation: sample movies
        # (batch, seq_len, vocab_size + n_movies)
        if self.sample_movies:
            # Prepare probability vector for sampling
            movie_probabilities = F.softmax(movie_recommendations.view(-1, n_movies), dim=1)
            # zero out the forbidden movies
            if forbid_movies is not None:
                if batch_size > 1:
                    raise ValueError("forbid_movies functionality only implemented with batch_size=1 for now")
                for movieId in forbid_movies:
                    movie_probabilities[:, movieId] = 0
            # Sample movies
            sampled_movies = movie_probabilities.multinomial(1).view(
                batch_size, max_utterance_length).data.cpu().numpy()
            # Fill a new recommendations vector with sampled movies
            movie_recommendations = torch.zeros(batch_size, max_utterance_length, n_movies, device=self.device)
            for i in range(batch_size):
                for j in range(max_utterance_length):
                    # compensate bias towards sampled movie by putting the maximum probability of a movie instead of 1
                    movie_recommendations[i, j, sampled_movies[i, j]] = max_probabilities[i]

            output = torch.cat((
                switch.sigmoid() * F.softmax(language_output / temperature, dim=-1),
                (-switch).sigmoid() * movie_recommendations
            ), dim=2)
        else:
            output = torch.cat((
                F.logsigmoid(switch) + F.log_softmax(language_output / temperature, dim=-1),
                F.logsigmoid(-switch) + F.log_softmax(movie_recommendations / temperature, dim=-1)
            ), dim=2)
        return output

    def forward(self, rec_inputs, gen_inputs, forbid_movies):
        """
        Forward pass through the SwitchDecodePipeline.

        Args:
            rec_inputs (dict): Recommendation inputs.
            gen_inputs (dict): Generation inputs.
            forbid_movies (set): Set of forbidden movie indices.

        Returns:
            torch.Tensor: Forward pass result.
        """
        batch_size, max_conv_length, max_utterance_length = gen_inputs['dialogue'].shape[:3]
        recs = self.recs if self.recs is not None else self.rec_module.forward(**rec_inputs) # (batch, seq_len, n_movies)
        pad_tensor = torch.zeros(batch_size, 1, recs.shape[-1], device=recs.device)
        recs = torch.cat((pad_tensor, recs), 1)[:, -2, :]
        context_, hidden, language_output = self.gen_module.forward(**gen_inputs, context=self.context)
        context = self.context if self.context is not None else context_
        # merge batch_size and max_conv_length into one dimension for switch

        context = context.expand(batch_size*max_conv_length, max_utterance_length, self.context_size).contiguous()
        hidden = hidden.view(batch_size*max_conv_length, max_utterance_length, -1)
        switch_input = torch.cat((context, hidden), dim=-1)
        # FIXME: the conv_length dimension is not unrolled
        language_output = language_output.view(batch_size*max_conv_length, max_utterance_length, -1)
        output = self.switch_decode(switch_input, language_output, recs, temperature=self.t, forbid_movies=forbid_movies)

        return output.view(batch_size, max_conv_length, max_utterance_length, -1)

    def replace_movie_with_words(self, tokens):
        """
        If the ID corresponds to a movie, returns the sequence of tokens that correspond to this movie name
        Args:
            tokens (list): List of token IDs.
        Returns:
            list: Modified list of tokens.
        """
        res = []
        for token_id in tokens:
            if token_id <= self.vocab_size:
                res.append(token_id)
            else:
                movie_name = self.rec_tokenizer.decode(token_id - self.vocab_size)
                res.extend(self.gen_tokenizer.encode(movie_name, add_special_tokens=False))
        return res

    def response(self, query: str, beam_size=10, forbid_movies=None, temperature=1, **kwargs):
        """
        Generate a response using the SwitchDecodePipeline.

        Args:
            query (str): The input query for generating a response.
            beam_size (int, optional): Beam size for response generation. Defaults to 10.
            forbid_movies (set, optional): Set of forbidden movie indices. Defaults to None.
            temperature (float, optional): Temperature parameter for controlling randomness. Defaults to 1.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated response.
        """
        if forbid_movies is None:
            forbid_movies = set()
        self.t = temperature
        initial_sequence = [self.gen_tokenizer.tokenizers[1].bos_token_id]
        Beam.end_token = self.gen_tokenizer.tokenizers[1].eos_token_id
        beams = BeamSearch.initial_beams(initial_sequence)
        max_conv_length = len(re.findall(SEP_TOKEN, query)) + 1
        conv_repr = self.gen_module.response(query, self.gen_tokenizer, encoder_only=True)
        # TODO: if first utterance is recommender, add a 0-context at the beginning
        self.context = conv_repr[:, -1:, :]
        self.recs = self.rec_module.response(query, self.rec_tokenizer)

        for i in range(self.config.max_seq_length):
            # compute probabilities for each beam
            probabilities = []
            for beam in beams:
                # add batch_dimension
                gen_inputs = {
                    'dialogue': torch.tensor(beam.sequence, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0),
                    'lengths': torch.tensor([len(beam.sequence)]).unsqueeze(0),
                }
                beam_forbidden_movies = forbid_movies.union(beam.mentioned_movies)
                prob = self.forward(
                    rec_inputs=None,
                    gen_inputs=gen_inputs,
                    forbid_movies=beam_forbidden_movies,
                )
                # get probabilities for the next token to generate
                probabilities.append(prob[0, 0, -1, :].cpu())
            # update beams
            beams = BeamSearch.update_beams(beams, beam_size, probabilities)
            # replace movie names with the corresponding words
            for beam in beams:
                if beam.sequence[-1] > self.vocab_size:
                    # update the list of movies mentioned for preventing repeated recommendations
                    beam.mentioned_movies.add(beam.sequence[-1] -self.vocab_size)
                    beam.sequence[-1:] = self.replace_movie_with_words(beam.sequence[-1])
        best_beam = get_best_beam(beams)
        return self.gen_tokenizer.batch_decode(best_beam.sequence)