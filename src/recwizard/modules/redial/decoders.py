import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .beam_search import BeamSearch, Beam, get_best_beam

class DecoderGRU(nn.Module):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    def __init__(self, hidden_size, context_size, num_layers, word_embedding, peephole):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        # peephole: concatenate the context to the input at every time step
        self.peephole = peephole
        if not peephole and context_size != hidden_size:
            raise ValueError("peephole=False: the context size {} must match the hidden size {} in DecoderGRU".format(
                context_size, hidden_size))
        self.embedding = word_embedding
        self.gru = nn.GRU(
            input_size=self.embedding.embedding_dim + context_size * self.peephole,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, input_sequence, lengths, context=None, state=None):
        """
        If not peephole, use the context vector as initial hidden state at each layer.
        If peephole, concatenate context to embeddings at each time step instead.
        If context is not provided, assume that a state is given (for generation)

        Args:
            input_sequence: (batch_size, seq_len)
            lengths: (batch_size)
            context: (batch, hidden_size) vector on which to condition
            state: (batch, num_layers, hidden_size) gru state
        Returns:
            ouptut predictions (batch_size, seq_len, hidden_size) [, h_n (batch, num_layers, hidden_size)]
        """
        embedded = self.embedding(input_sequence)
        if context is not None:
            batch_size, context_size = context.data.shape
            seq_len = input_sequence.data.shape[1]
            if self.peephole:
                context_for_input = context.unsqueeze(1).expand(batch_size, seq_len, context_size)
                embedded = torch.cat((embedded, context_for_input), dim=2)
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)

            if not self.peephole:
                # No peephole. Use context as initial hidden state
                # expand to the number of layers in the decoder
                context = context.unsqueeze(0).expand(
                    self.num_layers, batch_size, self.hidden_size).contiguous()

                output, _ = self.gru(packed, context)
            else:
                output, _ = self.gru(packed)
            return pad_packed_sequence(output, batch_first=True)[0]
        elif state is not None:
            output, h_n = self.gru(embedded, state)
            return output, h_n
        else:
            raise ValueError("Must provide at least state or context")


class SwitchingDecoder(nn.Module):
    """
    Decoder that takes the recommendations into account.
    A switch choses whether to output a movie or a word
    """

    def __init__(self,
                 hidden_size,
                 context_size,
                 num_layers,
                 peephole,
                 word_embedding=None,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.peephole = peephole
        self.vocab_size = word_embedding.num_embeddings
        self.decoder = DecoderGRU(
            hidden_size=self.hidden_size,
            context_size=self.context_size,
            num_layers=self.num_layers,
            word_embedding=word_embedding,
            peephole=self.peephole
        )
        self.language_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.switch = nn.Linear(in_features=self.hidden_size + self.context_size, out_features=1)
        # if self.cuda_available:
        #     self.cuda()

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.decoder.set_pretrained_embeddings(embedding_matrix)

    def forward(self, input, lengths, context, movie_recommendations, log_probabilities, sample_movies,
                forbid_movies=None, temperature=1):
        """

        Args:
            input: (batch, max_utterance_length)
            log_probabilities:
            temperature:
            lengths:
            context: (batch, hidden_size)
            movie_recommendations: (batch, n_movies) the movie recommendations that condition the utterances.
            Not necessarily in [0,1] range
            sample_movies: (for generation) If true, sample a movie for each utterance, returning one-hot vectors
            forbid_movies: (for generation) If provided, specifies movies that cannot be sampled
        Returns:
            [log] probabilities (batch, max_utterance_length, vocab + n_movies)
        """
        batch_size, max_utterance_length = input.data.shape[:2]
        # Run language decoder
        # (batch, seq_len, hidden_size)
        hidden = self.decoder(input, lengths, context=context)
        # (batch, seq_len, vocab_size)
        language_output = self.language_out(hidden)

        # used in sampling
        max_probabilities, _ = torch.max(F.softmax(movie_recommendations, dim=1), dim=1)
        # expand context and movie_recommendations to each time step
        context = context.unsqueeze(1).expand(
            batch_size, max_utterance_length, self.context_size).contiguous()
        n_movies = movie_recommendations.data.shape[1]
        movie_recommendations = movie_recommendations.unsqueeze(1).expand(
            batch_size, max_utterance_length, n_movies).contiguous()

        # Compute Switch
        # (batch, seq_len, 2 * hidden_size)
        switch_input = torch.cat((context, hidden), dim=2)
        switch = self.switch(switch_input)

        # For generation: sample movies
        # (batch, seq_len, vocab_size + n_movies)
        if sample_movies:
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
            movie_recommendations = torch.zeros(batch_size, max_utterance_length, n_movies, device=input.device)
            for i in range(batch_size):
                for j in range(max_utterance_length):
                    # compensate bias towards sampled movie by putting the maximum probability of a movie instead of 1
                    movie_recommendations[i, j, sampled_movies[i, j]] = max_probabilities[i]

            if log_probabilities:
                raise ValueError("Sample movies only works with log_probabilities=False for now.")
                # output = torch.cat((
                #     F.logsigmoid(switch) + F.log_softmax(language_output / temperature, dim=2),
                #     F.logsigmoid(-switch) + torch.log(movie_recommendations)
                # ), dim=2)
            else:
                output = torch.cat((
                    switch.sigmoid() * F.softmax(language_output / temperature, dim=2),
                    (-switch).sigmoid() * movie_recommendations
                ), dim=2)
            return output
        if log_probabilities:
            output = torch.cat((
                F.logsigmoid(switch) + F.log_softmax(language_output / temperature, dim=2),
                F.logsigmoid(-switch) + F.log_softmax(movie_recommendations / temperature, dim=2)
            ), dim=2)
        else:
            output = torch.cat((
                switch.sigmoid() * F.softmax(language_output / temperature, dim=2),
                (-switch).sigmoid() * F.softmax(movie_recommendations / temperature, dim=2)
            ), dim=2)
        return output

    def replace_movie_with_words(self, tokens, tokenizer):
        """
        If the ID corresponds to a movie, returns the sequence of tokens that correspond to this movie name
        Args:
            tokens: list of token ids
            tokenizer: tokenizer used to encode the movie names
        Returns: modified sequence
        """
        res = []
        for token_id in tokens:
            if token_id <= self.vocab_size:
                res.append(token_id)
            else:
                movie_name = tokenizer.decode([token_id])
                res.extend(tokenizer.tokenizers[1].encode(movie_name, add_special_tokens=False))
        return res
    
    @property
    def device(self):
        return self.switch.weight.device

    def generate(self, initial_sequence=None, tokenizer=None, beam_size=10, max_seq_length=50, temperature=1,
                 forbid_movies=None, **kwargs):
        """
        Beam search sentence generation
        Args:
            initial_sequence: list giving the initial sequence of tokens
            kwargs: additional parameters to pass to model forward pass (e.g. a conditioning context)
        Returns:
            The best beam
        """
        if initial_sequence is None:
            initial_sequence = []
        initial_sequence = [tokenizer.bos_token_id] + initial_sequence
        Beam.end_token = tokenizer.eos_token_id
        beams = BeamSearch.initial_beams(initial_sequence)
        for i in range(max_seq_length):
            # compute probabilities for each beam
            probabilities = []
            for beam in beams:
                # add batch_dimension
                model_input = torch.tensor(beam.sequence, device=self.device, dtype=torch.long).unsqueeze(0)
                beam_forbidden_movies = forbid_movies.union(beam.mentioned_movies)
                prob = self.forward(
                    input=model_input,
                    lengths=torch.tensor([len(beam.sequence)], device=self.device),
                    log_probabilities=False,
                    forbid_movies=beam_forbidden_movies,
                    temperature=temperature,
                    **kwargs
                )
                # get probabilities for the next token to generate
                probabilities.append(prob[0, -1, :].cpu())
            # update beams
            beams = BeamSearch.update_beams(beams, beam_size, probabilities)
            # replace movie names with the corresponding words
            for beam in beams:
                if beam.sequence[-1] > self.vocab_size:
                    # update the list of movies mentioned for preventing repeated recommendations
                    beam.mentioned_movies.add(beam.sequence[-1] - self.vocab_size)
                    beam.sequence[-1:] = self.replace_movie_with_words(beam.sequence[-1:], tokenizer)

        return get_best_beam(beams)
