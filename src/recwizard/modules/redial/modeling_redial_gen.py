import torch
from torch import nn as nn
import re

from recwizard import BaseModule
from recwizard.modules.monitor import monitor
from recwizard.utility import SEP_TOKEN, DeviceManager
from .utils import sort_for_packed_sequence
from .configuration_redial_gen import RedialGenConfig
from .hrnn import HRNN
from .utils import preprocess, fill_movie_occurrences
from .decoders import SwitchingDecoder

class RedialGen(BaseModule):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    config_class = RedialGenConfig
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
        
        movies = [m[0] for m in re.findall(r'<entity>(.*?)</entity>', raw_input)]
        texts = raw_input.split(SEP_TOKEN)
        batch_text, senders = zip(*[preprocess(text) for text in texts])
        encodings = tokenizer(batch_text, padding=True, return_tensors='pt', truncation=True)

        movie_occurrences = torch.stack([fill_movie_occurrences(encodings['sen_encoding'], batch_text, movie_name=movie) for movie in movies]) if len(movies) > 0 else torch.zeros((0, 0, 1))
        input_ids = encodings['sen_encoding']['input_ids']
        attention_mask = encodings['sen_encoding']['attention_mask']
        gen_input = {
            'dialogue': encodings['dialogue']['input_ids'],
            'lengths': torch.sum(attention_mask, dim=-1),
            # hrnn_input
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'senders': torch.as_tensor(senders),
            'movie_occurrences': movie_occurrences,
            'conversation_lengths': torch.tensor(len(texts)),
        }
        gen_input = {k: v.unsqueeze(0) for k, v in gen_input.items()}
        gen_input = DeviceManager.copy_to_device(gen_input, device=self.device)
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
        best_beam = self.decoder.generate(**inputs, tokenizer=tokenizer.tokenizers['dialogue'])
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

