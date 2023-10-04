import torch

from recwizard import BaseModule
from .configuration_redial_rec import RedialRecConfig
from .tokenizer_redial_rec import RedialRecTokenizer
from .hrnn_for_classification import HRNNForClassification
from .autorec import AutoRec
from recwizard.utility import WrapSingleInput, DeviceManager
from recwizard.modules.monitor import monitor
import torch


class RedialRec(BaseModule):
    config_class = RedialRecConfig
    tokenizer_class = RedialRecTokenizer
    LOAD_SAVE_IGNORES = {"sentiment_analysis.encoder.base_encoder", }

    def __init__(self, config: RedialRecConfig, recommend_new_movies=True, **kwargs):
        super().__init__(config, **kwargs)
        self.n_movies = config.n_movies
        self.recommend_new_movies = recommend_new_movies
        self.sentiment_analysis = HRNNForClassification(**config.sa_params)
        self.recommender = AutoRec(**config.autorec_params, n_movies=self.n_movies)


        # freeze sentiment analysis
        for param in self.sentiment_analysis.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids,
                attention_mask,
                senders,
                movieIds,
                conversation_lengths,
                movie_occurrences,
                **kwargs
                ):
        """
        Args:
            input_ids: (batch, max_conv_length, max_utt_length)
            attention_mask: (batch, max_conv_length, max_utt_length)
            senders: (batch, max_conv_length)
            movieIds: (batch, max_conv_length, max_n_movies)
            conversation_lengths: (batch)
            movie_occurrences: (batch, max_conv_length, max_utterance_length)
            **kwargs:

        Returns:
            (batch_size, max_conv_length, n_movies_total) movie preferences
        """

        if movieIds.numel() > 0:
            i_liked = self.sentiment_analysis(input_ids, attention_mask, senders, movie_occurrences,
                                              conversation_lengths)['i_liked']
        else:
            i_liked = torch.zeros(input_ids.shape[0], 0, self.n_movies, device=self.device)

        batch_size, max_conv_length = input_ids.shape[:2]
        indices = [(i, j)  # i-th conversation in the batch, j-th movie in the conversation
                   for (i, conv_movie_occurrences) in enumerate(movie_occurrences)
                   for j in range(len(conv_movie_occurrences))
                   ]
        autorec_input = torch.zeros(batch_size, max_conv_length, self.n_movies, device=self.device)
        for i in range(batch_size):
            for j in range(len(movieIds[i])):
                mask = movie_occurrences[i][j].sum(dim=1).cumsum(dim=0) > 0
                autorec_input[i, :len(mask), movieIds[i][j]] = mask * i_liked[i][j][:len(mask)]

        recs = self.recommender(autorec_input, additional_context=None, range01=False)
        # The default recommender is not language_aware_recommender. So we omit the additional_context

        if self.recommend_new_movies:
            for batchId, j in indices:
                # (max_conv_length) mask that zeros out once the movie has been mentioned
                mask = torch.sum(movie_occurrences[batchId][j], dim=1) > 0
                # mask = mask.cumsum(dim=0) == 0
                # recs[batchId, :, movieIds[batchId][j]] = mask * recs[batchId, :, movieIds[batchId][j]]
                # FIXED: the recs are negative, applying zero mask is raising the probability!
                mask = mask.cumsum(dim=0) > 0
                recs[batchId, :, movieIds[batchId][j]] -= mask * 1e10
        return recs

    @WrapSingleInput
    @monitor
    def response(self, raw_input, tokenizer, return_dict=False, topk=3, **kwargs):
        tokenized_input = DeviceManager.copy_to_device(tokenizer(raw_input).data)
        logits = self.forward(**tokenized_input)
        movieIds = logits.topk(k=topk, dim=-1).indices[:, -1,
                   :].tolist()  # selects the recommendation for the last sentence
        output = tokenizer.batch_decode(movieIds)
        if return_dict:
            return {
                'logits': logits,
                'movieIds': movieIds,
                'output': output,
            }
        return output
