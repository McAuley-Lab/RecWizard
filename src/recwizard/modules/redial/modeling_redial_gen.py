import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from recwizard import BaseModule, monitor
from recwizard.utility import sort_for_packed_sequence, DeviceManager
from .beam_search import BeamSearch, Beam, get_best_beam
from .configuration_redial_gen import RedialGenConfig
from .tokenizer_redial_gen import RedialGenTokenizer
from .hrnn import HRNN


class RedialGen(BaseModule):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    config_class = RedialGenConfig
    tokenizer_class = RedialGenTokenizer
    LOAD_SAVE_IGNORES = {"encoder.base_encoder", }

    def __init__(self, config: RedialGenConfig, word_embedding=None, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = HRNN(**config.hrnn_params)
        word_embedding = nn.Embedding.from_pretrained(self.prepare_weight(word_embedding, 'decoder.embedding'))
        DeviceManager.device = self.device
        self.decoder = SwitchingDecoder(word_embedding=word_embedding, **config.decoder_params).to(self.device)
        self.n_movies = config.n_movies

    def forward(self, dialogue, lengths, recs, context=None, state=None, encode_only=False, **hrnn_input):
        """

        Args:
            dialogue: (batch_size, seq_len)
            lengths: (batch_size)
            recs: (batch_size, n_movies)
            context:
            state:
            encode_only:
            **hrnn_input:

        Returns:

        """
        batch_size, max_conv_length = dialogue.shape[:2]
        # only generate new context when context is None
        conv_repr = context if context is not None else self.encoder(**hrnn_input)['conv_repr']  # (batch, hidden_size)
        if encode_only:
            return conv_repr
        sorted_utterances, sorted_lengths, conv_repr, recs, rev, num_positive_lengths \
            = self.prepare_input_for_decoder(
            dialogue, lengths, conv_repr, recs
        )

        # Run decoder
        outputs = self.decoder(
            sorted_utterances,
            sorted_lengths,
            conv_repr,
            recs,
            log_probabilities=True,
            sample_movies=False
        )
        max_utterance_length = outputs.shape[1]
        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conv_length:
            pad_tensor = torch.zeros(
                batch_size * max_conv_length - num_positive_lengths,
                max_utterance_length,
                outputs.shape[-1],
                device=self.device
            )
            outputs = torch.cat((outputs, pad_tensor), dim=0)

        # retrieve original order
        outputs = outputs.index_select(0, rev). \
            view(batch_size, max_conv_length, max_utterance_length, -1)
        # (batch, max_conv_len, max_sentence_len, vocab + n_movie)
        return outputs

    @monitor
    def response(self, raw_input, tokenizer, recs=None, return_dict=False, beam_size=10, forbid_movies=None,
                 temperature=1, max_seq_length=40, **kwargs):
        gen_input = DeviceManager.copy_to_device(tokenizer([raw_input]).data, device=self.device)
        context = self.forward(**gen_input, recs=recs, encode_only=True)[0]
        if recs is None:
            recs = torch.zeros(1, self.n_movies, device=self.device)
        mentioned_movies = forbid_movies if forbid_movies is not None else set()

        inputs = {
            "initial_sequence": [],
            "forbid_movies": mentioned_movies,
            "beam_size": beam_size,
            "max_seq_length": max_seq_length,
            "movie_recommendations": recs[-1].unsqueeze(0),
            "context": context[-1].unsqueeze(0),
            "sample_movies": False,
            "temperature": temperature
        }
        best_beam = self.decoder.generate(**inputs, tokenizer=tokenizer)
        mentioned_movies.update(best_beam.mentioned_movies)
        output = tokenizer.decode(best_beam.sequence, skip_special_tokens=True)
        if return_dict:
            return {
                "output": output,
                "movieIds": list(mentioned_movies)
                # These movieIds could be passed as forbid_movies in the following queries
            }
        return output

    def prepare_input_for_decoder(self, dialogue, lengths, conv_repr, recs):
        batch_size, max_conv_length = dialogue.shape[:2]
        utterances = dialogue.view(batch_size * max_conv_length, -1)

        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths)

        sorted_utterances = utterances.index_select(0, sorted_idx)

        # shift the context vectors one step in time
        pad_tensor = torch.zeros(
            batch_size, 1, int(self.config.hrnn_params['conversation_encoder_hidden_size']), device=self.device)
        conv_repr = torch.cat((pad_tensor, conv_repr), 1).narrow(
            1, 0, max_conv_length)
        # and reshape+reorder the same way as utterances
        conv_repr = conv_repr.contiguous().view(
            batch_size * max_conv_length, self.config.hrnn_params['conversation_encoder_hidden_size']) \
            .index_select(0, sorted_idx)

        # shift the movie recommendations one step in time
        pad_tensor = torch.zeros(batch_size, 1, self.n_movies, device=self.device)
        recs = torch.cat((pad_tensor, recs), 1).narrow(
            1, 0, max_conv_length)
        # and reshape+reorder movie_recommendations the same way as utterances
        recs = recs.contiguous().view(
            batch_size * max_conv_length, -1).index_select(0, sorted_idx)

        # consider only lengths > 0
        num_positive_lengths = torch.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conv_repr = conv_repr[:num_positive_lengths]
        recs = recs[:num_positive_lengths]

        return sorted_utterances, sorted_lengths, conv_repr, recs, rev, num_positive_lengths


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
            movie_recommendations = torch.zeros(batch_size, max_utterance_length, n_movies, device=DeviceManager.device)
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
                model_input = torch.tensor(beam.sequence, device=DeviceManager.device, dtype=torch.long).unsqueeze(0)
                beam_forbidden_movies = forbid_movies.union(beam.mentioned_movies)
                prob = self.forward(
                    input=model_input,
                    lengths=torch.tensor([len(beam.sequence)], device=DeviceManager.device),
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
