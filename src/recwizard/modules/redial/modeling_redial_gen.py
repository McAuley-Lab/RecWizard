import torch
from torch import nn as nn
import re

from recwizard import BaseModule
from recwizard import monitor
from recwizard.utils import SEP_TOKEN, create_item_list
from recwizard.modules.redial.configuration_redial_gen import RedialGenConfig
from recwizard.modules.redial.tokenizer_redial_gen import RedialGenTokenizer

from recwizard.modules.redial.original_utils import sort_for_packed_sequence
from recwizard.modules.redial.original_hrnn import HRNN
from recwizard.modules.redial.original_utils import preprocess, fill_movie_occurrences
from recwizard.modules.redial.original_decoders import SwitchingDecoder


class RedialGen(BaseModule):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    config_class = RedialGenConfig
    tokenizer_class = RedialGenTokenizer
    LOAD_SAVE_IGNORES = {
        "encoder.base_encoder",
    }

    def __init__(self, config: RedialGenConfig, word_embedding=None, **kwargs):
        super().__init__(config, **kwargs)
        self.encoder = HRNN(**config.hrnn_params)
        word_embedding = nn.Embedding.from_pretrained(self.prepare_weight(word_embedding, "decoder.embedding"))
        self.decoder = SwitchingDecoder(word_embedding=word_embedding, **config.decoder_params)
        self.n_movies = config.n_movies

    def forward(self, dialogue, lengths, rec_logits, context=None, state=None, encode_only=False, **hrnn_input):
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
        conv_repr = context if context is not None else self.encoder(**hrnn_input)["conv_repr"]  # (batch, hidden_size)
        if encode_only:
            return conv_repr
        sorted_utterances, sorted_lengths, conv_repr, rec_logits, rev, num_positive_lengths = (
            self.prepare_input_for_decoder(dialogue, lengths, conv_repr, rec_logits)
        )

        # Run decoder
        outputs = self.decoder(
            sorted_utterances, sorted_lengths, conv_repr, rec_logits, log_probabilities=True, sample_movies=False
        )
        max_utterance_length = outputs.shape[1]
        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conv_length:
            pad_tensor = torch.zeros(
                batch_size * max_conv_length - num_positive_lengths,
                max_utterance_length,
                outputs.shape[-1],
                device=self.device,
            )
            outputs = torch.cat((outputs, pad_tensor), dim=0)

        # retrieve original order
        outputs = outputs.index_select(0, rev).view(batch_size, max_conv_length, max_utterance_length, -1)
        # (batch, max_conv_len, max_sentence_len, vocab + n_movie)
        return outputs

    @monitor
    def response(
        self,
        raw_input,
        tokenizer,
        rec_logits=None,
        return_dict=False,
        beam_size=10,
        forbid_movies=None,
        temperature=1,
        max_seq_length=40,
        **kwargs,
    ):
        # Chat message to model inputs
        # ReDIAL is using an internal input preparation function, which is a customized process for the model
        inputs = self._tokenize_input(raw_input, tokenizer)

        # Model generates (switching decoder is used here)
        best_beam, mentioned_movies = self.generate(
            inputs,
            rec_logits=rec_logits,
            tokenizer=tokenizer,
            beam_size=beam_size,
            forbid_movies=forbid_movies,
            temperature=temperature,
            max_seq_length=max_seq_length,
        )
        output = tokenizer.decode(best_beam.sequence, skip_special_tokens=True)

        # Return the output
        if return_dict:
            return {
                "output": output,
                "item_ids": list(mentioned_movies),
                # These movie_ids could be passed as forbid_movies in the following queries
            }
        return output

    def generate(
        self,
        inputs,
        rec_logits=None,
        tokenizer=None,
        beam_size=10,
        forbid_movies=None,
        temperature=1,
        max_seq_length=40,
        **kwargs,
    ):
        context = self.forward(**inputs, rec_logits=rec_logits, encode_only=True)[0]
        if rec_logits is None:
            rec_logits = torch.zeros(1, self.n_movies, device=self.device)
        mentioned_movies = forbid_movies if forbid_movies is not None else set()
        best_beam = self.decoder.generate(
            initial_sequence=[],
            tokenizer=tokenizer,
            forbid_movies=mentioned_movies,
            beam_size=beam_size,
            max_seq_length=max_seq_length,
            movie_recommendations=rec_logits[-1].unsqueeze(0),
            context=context[-1].unsqueeze(0),
            sample_movies=False,
            temperature=temperature,
        )
        mentioned_movies.update(best_beam.mentioned_movies)

        return best_beam, mentioned_movies

    def _tokenize_input(self, input, tokenizer):
        movies = create_item_list(input)
        texts = input.split(SEP_TOKEN)
        batch_text, senders = zip(*[preprocess(text) for text in texts])
        encodings = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True).to(self.device)

        movie_occurrences = (
            torch.stack(
                [fill_movie_occurrences(encodings["sen_encoding"], batch_text, movie_name=movie) for movie in movies]
            )
            if len(movies) > 0
            else torch.zeros((0, 0, 1))
        )
        input_ids = encodings["sen_encoding"]["input_ids"]
        attention_mask = encodings["sen_encoding"]["attention_mask"]
        gen_input = {
            "dialogue": encodings["dialogue"]["input_ids"],
            "lengths": torch.sum(attention_mask, dim=-1),
            # hrnn_input
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "senders": torch.as_tensor(senders),
            "movie_occurrences": movie_occurrences,
            "conversation_lengths": torch.tensor(len(texts)),
        }
        return {k: v.unsqueeze(0) for k, v in gen_input.items()}

    def prepare_input_for_decoder(self, dialogue, lengths, conv_repr, recs):
        batch_size, max_conv_length = dialogue.shape[:2]
        utterances = dialogue.view(batch_size * max_conv_length, -1)

        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths)

        sorted_utterances = utterances.index_select(0, sorted_idx)

        # shift the context vectors one step in time
        pad_tensor = torch.zeros(
            batch_size, 1, int(self.config.hrnn_params["conversation_encoder_hidden_size"]), device=self.device
        )
        conv_repr = torch.cat((pad_tensor, conv_repr), 1).narrow(1, 0, max_conv_length)
        # and reshape+reorder the same way as utterances
        conv_repr = (
            conv_repr.contiguous()
            .view(batch_size * max_conv_length, self.config.hrnn_params["conversation_encoder_hidden_size"])
            .index_select(0, sorted_idx)
        )

        # shift the movie recommendations one step in time
        pad_tensor = torch.zeros(batch_size, 1, self.n_movies, device=recs.device)
        recs = torch.cat((pad_tensor, recs), 1).narrow(1, 0, max_conv_length)
        # and reshape+reorder movie_recommendations the same way as utterances
        recs = recs.contiguous().view(batch_size * max_conv_length, -1).index_select(0, sorted_idx)

        # consider only lengths > 0
        num_positive_lengths = torch.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conv_repr = conv_repr[:num_positive_lengths]
        recs = recs[:num_positive_lengths]

        return sorted_utterances, sorted_lengths, conv_repr, recs, rev, num_positive_lengths
