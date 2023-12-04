import torch
import torch.nn as nn
import torch.nn.functional as F

from recwizard.utility import pad_and_stack
from recwizard import BaseConfig, BaseModule
from .hrnn import HRNN

class HRNNForClassification(BaseModule):

    def __init__(self, hrnn_params, output_classes, return_liked_probability=True, multiple_items_per_example=True):
        """
        Hierarchical Recurrent Neural Network for Classification.

        Args:
            hrnn_params (dict): Dictionary of HRNN parameters.
            output_classes (dict): Dictionary mapping output names to the number of classes for each output.
            return_liked_probability (bool, optional): Whether to return the liked probability (default: True).
            multiple_items_per_example (bool, optional): Should be set to True when each conversation corresponds to an example (e.g., when generating output).
                Should be set to False during training because each item corresponds to an example.

        Attributes:
            encoder (HRNN): The HRNN encoder module.
            linears (nn.ModuleDict): Dictionary of output linear layers for each output.
            return_liked_probability (bool): Whether to return the liked probability.
            multiple_items_per_example (bool): Whether each conversation corresponds to an example.

        """
        super().__init__(BaseConfig())
        self.encoder = HRNN(return_all=return_liked_probability, **hrnn_params)
        conv_bidrectional = hrnn_params['conv_bidirectional']
        encoder_output_size = hrnn_params['conversation_encoder_hidden_size']

        self.linears = nn.ModuleDict(
            {output: nn.Linear((1 + conv_bidrectional) * encoder_output_size, num_classes)
             for output, num_classes in output_classes.items()})

        self.return_liked_probability = return_liked_probability
        self.multiple_items_per_example = multiple_items_per_example


    def on_pretrain_finished(self, requires_grad=False):
        self.requires_grad_(requires_grad)
        self.return_liked_probability = True
        self.multiple_items_per_example = True

    def forward(self, input_ids, attention_mask, senders, movie_occurrences,
                conversation_lengths):
        """
        Forward pass of the HRNNForClassification model.

        Args:
            input_ids (Tensor): Input token IDs of shape (batch_size, max_conv_length, max_utt_length).
            attention_mask (Tensor): Attention mask of shape (batch_size, max_conv_length, max_utt_length).
            senders (Tensor): Senders information of shape (batch_size, max_conv_length).
            movie_occurrences (Tensor): Movie occurrences information of shape
                (batch_size, max_conv_length, max_utt_length).
            conversation_lengths (Tensor): List of conversation lengths for each batch element.

        Returns:
            dict: A dictionary containing model outputs, which may include class probabilities
            or liked probabilities for each input item or conversation.

        """
        if self.multiple_items_per_example:
            batch_size = len(input_ids)
            # if we have mutiple items in one example, first flatten (expand) the examples to have only one item to classify per example
            indices = [(i, j)  # i-th conversation in the batch, j-th item in the conversation
                       for (i, conv_item_occurrences) in enumerate(movie_occurrences)
                       for j in range(len(conv_item_occurrences))
                       ]
            batch_indices = torch.tensor([i[0] for i in indices], device=input_ids.device)
            # flatten item occurrences to shape (total_num_mentions_in_batch, max_conv_length, max_utt_length)
            movie_occurrences = pad_and_stack([oc for conv_oc in movie_occurrences for oc in conv_oc])
            input_ids = input_ids.index_select(0, batch_indices)
            attention_mask = attention_mask.index_select(0, batch_indices)
            senders = senders.index_select(0, batch_indices)
            conversation_lengths = conversation_lengths.index_select(0, batch_indices)

        conv_repr = self.encoder(
            input_ids, attention_mask, senders, movie_occurrences, conversation_lengths,
        )['conv_repr']

        if self.multiple_items_per_example:
            mask = movie_occurrences.sum(dim=2) > 0
            mask = mask.cumsum(dim=1) > 0  # (total_num_mentions_in_batch, max_conv_length)
            conv_repr = conv_repr * mask.unsqueeze(-1)

        if self.return_liked_probability:
            # return the liked probability at each utterance in the dialogue
            # (batch_size, max_conv_length, hidden_size*num_directions)
            # return the probability of the "liked" class
            output = {"i_liked": F.softmax(self.linears["i_liked"](conv_repr), dim=-1)[:, :, 1]}
            # output = {"i_liked": F.softmax(self.Iliked(conv_repr), dim=-1)[:, :, 1]}
        else:
            output = {key: linear(conv_repr) for key, linear in self.linears.items()}
        if self.multiple_items_per_example:  # recover the batch shape from flattend output
            batch_output = {key: [[] for _ in range(batch_size)] for key in output.keys()}
            for key in output.keys():
                for idx, (i, j) in enumerate(indices):
                    batch_output[key][i].append(output[key][idx])
            output = batch_output

        return output


class RedialSentimentAnalysisLoss(nn.Module):
    def __init__(self, class_weight, use_targets):
        super().__init__()
        self.class_weight = class_weight
        # string that specifies which targets to consider and, if specified, the weights
        self.use_targets = use_targets
        if len(use_targets.split()) == 6:
            self.weights = [int(x) for x in use_targets.split()[:3]]
        else:
            self.weights = [1, 1, 1]
        self.suggested_criterion = nn.BCEWithLogitsLoss()
        self.seen_criterion = nn.CrossEntropyLoss()
        if self.class_weight and "liked" in self.class_weight:
            self.liked_criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.class_weight["liked"]))
        else:
            self.liked_criterion = nn.CrossEntropyLoss()
        # if torch.cuda.is_available():
        #     self.cuda()

    def forward(self, output, target):
        loss = 0
        if "suggested" in self.use_targets:
            loss += self.weights[0] * (self.suggested_criterion(output["i_suggested"].squeeze(1), target[:, 0].float())
                                       + self.suggested_criterion(output["r_suggested"].squeeze(1),
                                                                  target[:, 3].float()))
        if "seen" in self.use_targets:
            loss += self.weights[1] * (self.seen_criterion(output["i_seen"], target[:, 1])
                                       + self.seen_criterion(output["r_seen"], target[:, 4]))
        if "liked" in self.use_targets:
            loss += self.weights[2] * (self.liked_criterion(output["i_liked"], target[:, 2])
                                       + self.liked_criterion(output["r_liked"], target[:, 5]))
        return loss
