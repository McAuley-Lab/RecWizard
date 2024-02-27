import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

from .utils import sort_for_packed_sequence


class HRNN(nn.Module):
    """
    Hierarchical Recurrent Neural Network

    params.keys() ['use_gensen', 'use_movie_occurrences', 'sentence_encoder_hidden_size',
    'conversation_encoder_hidden_size', 'sentence_encoder_num_layers', 'conversation_encoder_num_layers', 'use_dropout',
    ['embedding_dimension']]

    Input: Input["dialogue"] (batch, max_conv_length, max_utterance_length) Long Tensor
           Input["senders"] (batch, max_conv_length) Float Tensor
           Input["lengths"] (batch, max_conv_length) list
           (optional) Input["movie_occurrences"] (batch, max_conv_length, max_utterance_length) for word occurence
                                                 (batch, max_conv_length) for sentence occurrence. Float Tensor
    """

    # LOAD_SAVE_IGNORES = ('base_encoder',)
    def __init__(self,
                 sentence_encoder_model,
                 sentence_encoder_hidden_size,
                 sentence_encoder_num_layers,
                 conversation_encoder_hidden_size,
                 conversation_encoder_num_layers,
                 use_movie_occurrences,
                 conv_bidirectional=False,
                 return_all=True,
                 return_sentence_representations=False,
                 use_dropout=False,
                 ):
        super().__init__()
        self.conv_bidirectional = conv_bidirectional
        self.return_all = return_all
        self.return_sentence_representations = return_sentence_representations
        self.base_encoder = AutoModel.from_pretrained(sentence_encoder_model)
        self.sentence_encoder_hidden_size = sentence_encoder_hidden_size
        self.sentence_encoder = nn.GRU(
            self.base_encoder.config.hidden_size + (use_movie_occurrences == "word"),
            hidden_size=sentence_encoder_hidden_size,
            num_layers=sentence_encoder_num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.conversation_encoder = nn.GRU(
            input_size=2 * sentence_encoder_hidden_size
                       + 1 + (use_movie_occurrences == "sentence"),
            # concatenation of 2 directions for sentence encoders + sender informations + movie occurences
            hidden_size=conversation_encoder_hidden_size,
            num_layers=conversation_encoder_num_layers,
            batch_first=True,
            bidirectional=conv_bidirectional
        )
        self.use_movie_occurrences = use_movie_occurrences
        self.dropout = nn.Dropout(p=use_dropout) if use_dropout else None

    def get_sentence_representations(self, input_ids, attention_mask, senders, movie_occurrences=None):
        lengths = attention_mask.sum(dim=-1)  # mark the length of tokenized sentences
        batch_size, max_conversation_length = len(lengths), len(lengths[0])
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths)

        # reshape and reorder

        # # consider sequences of length > 0 only
        num_positive_lengths = torch.sum(lengths > 0)
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        sorted_idx = sorted_idx[:num_positive_lengths]

        batch_encoding = {
            'input_ids': input_ids.view(batch_size * max_conversation_length, -1).index_select(0, sorted_idx),
            'attention_mask': attention_mask.view(batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
        }
        with torch.no_grad():
            embedded = self.base_encoder(**batch_encoding, output_hidden_states=True,
                                         return_dict=True).last_hidden_state

        # (< batch_size * max conversation_length, max_sentence_length, embedding_size/2048 for gensen)

        if self.dropout:
            embedded = self.dropout(embedded)

        if self.use_movie_occurrences == "word":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            # reshape and reorder movie occurrences by utterance length
            movie_occurrences = movie_occurrences.view(
                batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
            # keep indices where sequence_length > 0
            embedded = torch.cat((embedded, movie_occurrences.unsqueeze(2)), 2)

        packed_sentences = pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)
        # Apply encoder and get the final hidden states
        _, sentence_representations = self.sentence_encoder(packed_sentences)
        # (2*num_layers, < batch_size * max_conv_length, hidden_size)
        # Concat the hidden states of the last layer (two directions of the GRU)
        sentence_representations = torch.cat((sentence_representations[-1], sentence_representations[-2]),
                                             1)  # (num_positive_lengths , 2*hidden_size)

        if self.dropout:
            sentence_representations = self.dropout(sentence_representations)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            pad_tensor = torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                2 * self.sentence_encoder_hidden_size,
                device=sentence_representations.device
            )
            sentence_representations = torch.cat((
                sentence_representations,
                pad_tensor
            ), 0)
        # sentence_representations.data.shape = (batch_size * max_conversation_length, 2*hidden_size)
        # Retrieve original sentence order and Reshape to separate conversations
        sentence_representations = sentence_representations.index_select(0, rev).view(
            batch_size,
            max_conversation_length,
            2 * self.sentence_encoder_hidden_size
        )
        # Append sender information
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        # Append movie occurrence information if required
        if self.use_movie_occurrences == "sentence":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            sentence_representations = torch.cat(
                (sentence_representations, torch.sign(torch.sum(movie_occurrences, dim=-1, keepdim=True))), 2)
        #  (batch_size, max_conv_length, 513 + self.params['use_movie_occurrences'])
        return sentence_representations

    def forward(self, input_ids, attention_mask, senders, movie_occurrences, conversation_lengths, **kwargs):
        # get sentence representations
        sentence_representations = self.get_sentence_representations(input_ids, attention_mask, senders,
                                                                     movie_occurrences)
        # (batch_size, max_conv_length, 2*sent_hidden_size + 1 + use_movie_occurences)
        # Pass whole conversation into GRU

        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(conversation_lengths)

        # reorder in decreasing sequence length
        sorted_representations = sentence_representations.index_select(0, sorted_idx)
        packed_sequences = pack_padded_sequence(sorted_representations, sorted_lengths.cpu(), batch_first=True)
        conversation_representations, last_state = self.conversation_encoder(packed_sequences)

        # retrieve original order
        conversation_representations, _ = pad_packed_sequence(conversation_representations, batch_first=True)
        conversation_representations = conversation_representations.index_select(0, rev)
        # (num_layers * num_directions, batch, conv_hidden_size)
        last_state = last_state.index_select(1, rev)
        if self.dropout:
            conversation_representations = self.dropout(conversation_representations)
            last_state = self.dropout(last_state)
        res = {}
        if self.return_all:
            if not self.return_sentence_representations:
                # return the last layer of the GRU for each t.
                # (batch_size, max_conv_length, hidden_size*num_directions
                res["conv_repr"] = conversation_representations
            else:
                # also return sentence representations
                res["conv_repr"] = conversation_representations
                res["sent_repr"] = sentence_representations
        else:
            # get the last hidden state only
            if self.conv_bidirectional:
                # Concat the hidden states for the last layer (two directions of the GRU)
                last_state = torch.cat((last_state[-1], last_state[-2]), 1)
                # (batch_size, num_directions*hidden_size)
                res["conv_repr"] = last_state
            else:
                # Return the hidden state from the last layers
                res["conv_repr"] = last_state[-1]
        return res
