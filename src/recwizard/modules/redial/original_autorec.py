import torch
import torch.nn as nn

from recwizard import BaseConfig, BaseModule
import logging

logger = logging.getLogger(__name__)


# Adapted from https://github.com/RaymondLi0/conversational-recommendations/blob/master/models/autorec.py

class AutoRec(BaseModule):  # We use features from BaseModule to load previous checkpoints
    """
    User-based Autoencoder for Collaborative Filtering
    """

    def __init__(self, n_movies, layer_sizes, g, f):
        super().__init__(BaseConfig())
        self.n_movies = n_movies
        self.layer_sizes = layer_sizes

        if g == 'identity':
            self.g = lambda x: x
        elif g == 'sigmoid':
            self.g = nn.Sigmoid()
        elif g == 'relu':
            self.g = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for g : {}".format(self.config.g))

        self.encoder = UserEncoder(layer_sizes=self.layer_sizes, n_movies=self.n_movies, f=f)
        self.user_representation_size = self.layer_sizes[-1]
        self.decoder = nn.Linear(in_features=self.user_representation_size, out_features=self.n_movies)

        # if checkpoint is not None:
        #     self.load_from_checkpoint(checkpoint)

        # if self.cuda_available:
        #     self.cuda()

    def load_checkpoint(self, checkpoint, verbose=True, strict=True, LOAD_PREFIX=''):
        if verbose:
            logger.info(f"AutoRec: loading model from {checkpoint}")
        # Load pretrained model, keeping only the first n_movies. (see merge_indexes function in match_movies)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # load all weights except for the weights of the first layer and the decoder
        model_dict.update({k: v for k, v in checkpoint.items()
                           if k.startswith(
                LOAD_PREFIX) and k != LOAD_PREFIX + "encoder.layers.0.weight" and "decoder" not in k})
        # load first layer and decoder: assume the movies to keep are the n_movies first movies
        encoder0weight = checkpoint[LOAD_PREFIX + "encoder.layers.0.weight"][:, :self.n_movies]
        decoderweight = checkpoint[LOAD_PREFIX + "decoder.weight"][:self.n_movies, :]
        decoderbias = checkpoint[LOAD_PREFIX + "decoder.bias"][:self.n_movies]
        # If checkpoint has fewer movies than the model, append zeros (no actual recommendations for new movies)
        # (When using an updated movie list)
        if encoder0weight.shape[1] < self.n_movies:
            encoder0weight = torch.cat((
                encoder0weight,
                torch.zeros(encoder0weight.shape[0], self.n_movies - encoder0weight.shape[1], device=self.device)),
                dim=1)
            decoderweight = torch.cat((
                decoderweight,
                torch.zeros(self.n_movies - decoderweight.shape[0], decoderweight.shape[1], device=self.device)),
                dim=0)
            decoderbias = torch.cat((
                decoderbias, torch.zeros(self.n_movies - decoderbias.shape[0], device=self.device)), dim=0)
        model_dict.update({
            LOAD_PREFIX + "encoder.layers.0.weight": encoder0weight,
            LOAD_PREFIX + "decoder.weight": decoderweight,
            LOAD_PREFIX + "decoder.bias": decoderbias,
        })
        self.load_state_dict(model_dict, strict=strict)
        if verbose:
            logger.info("AutoRec: finish loading")

    def forward(self, input, additional_context=None, range01=True):
        """

        :param input: (batch, n_movies)
        :param additional_context: potential information to add to user representation (batch, user_rep_size)
        :param range01: If true, apply sigmoid to the output
        :return: output recommendations (batch, n_movies)
        """
        # get user representation
        encoded = self.encoder(input, raw_last_layer=True)
        # eventually use additional context
        if additional_context is not None:
            encoded = self.encoder.f(encoded + additional_context)
        else:
            encoded = self.encoder.f(encoded)
        # decode to get movie recommendations
        if range01:
            return self.g(self.decoder(encoded))
        else:
            return self.decoder(encoded)



class UserEncoder(nn.Module):
    def __init__(self, layer_sizes, n_movies, f):
        """

        :param layer_sizes: list giving the size of each layer
        :param n_movies:
        :param f:
        """
        super(UserEncoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=n_movies, out_features=layer_sizes[0]) if i == 0
                                     else nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i])
                                     for i in range(len(layer_sizes))])

        if f == 'identity':
            self.f = lambda x: x
        elif f == 'sigmoid':
            self.f = nn.Sigmoid()
        elif f == 'relu':
            self.f = nn.ReLU()
        else:
            raise ValueError("Got invalid function name for f : {}".format(f))

    def forward(self, input, raw_last_layer=False):
        for (i, layer) in enumerate(self.layers):
            if raw_last_layer and i == len(self.layers) - 1:
                # do not apply activation to last layer
                input = layer(input)
            else:
                input = self.f(layer(input))
        return input




class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        # Sum of losses
        self.mse_loss = nn.MSELoss(size_average=False)
        # Keep track of number of observer targets to normalize loss
        self.nb_observed_targets = 0

    def forward(self, input, target):
        # only consider the observed targets
        mask = (target != -1)
        observed_input = torch.masked_select(input, mask)
        observed_target = torch.masked_select(target, mask)
        # increment nb of observed targets
        self.nb_observed_targets += len(observed_target)
        loss = self.mse_loss(observed_input, observed_target)
        return loss

    def normalize_loss_reset(self, loss):
        """
        returns loss divided by nb of observed targets, and reset nb_observed_targets to 0
        :param loss: Total summed loss
        :return: mean loss
        """
        if self.nb_observed_targets == 0:
            raise ValueError(
                "Nb observed targets was 0. Please evaluate some examples before calling normalize_loss_reset")
        n_loss = loss / self.nb_observed_targets
        self.nb_observed_targets = 0
        return n_loss
