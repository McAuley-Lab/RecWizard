import copy
import logging
import os
from collections import defaultdict, OrderedDict
from typing import Tuple, Callable, Optional, Iterable, List, Type, Union

import numpy as np
import torch
from torch.nn.modules.module import _IncompatibleKeys
from transformers import PreTrainedModel, PreTrainedTokenizer

from .modules.monitor import monitor
from .configuration_utils import BaseConfig


class BaseModule(PreTrainedModel):
    """
    The base class that's used for any modules that's based on nn.Module. Supporting partial load/save and state dict mapping.
    """

    LOAD_SAVE_IGNORES = set()
    """
    The set of external modules which will be skipped both in `state_dict()` and `load_state_dict()`.
        e.g. When a fixed PretrainedModel is used as an encoder, we can set ``LOAD_SAVE_IGNORES={"encoder"}`` to skip
        loading the encoder.
        
    .. tip::
        We suggest using it when your module has a fixed part, in order to reduce the size of checkpoint.
        
    """
    tokenizer_class: Type[PreTrainedTokenizer] = None
    """
    The tokenizer class that should be used for this module.
    """
    config_class = BaseConfig

    def __init__(self, config: BaseConfig = None, **kwargs):
        """

        Args:
            config: config for PreTrainedModel

        """
        super().__init__(config, **kwargs)
        self.model_name_or_path = None

    def get_tokenizer(self):
        """
        Serves as a default tokenizer getter for the module.
        Will load the tokenizer from `self.model_name_or_path` if it has been set.

        .. warning::
            This function will be called when a module is passed to a `BasePipeline` instance
            without providing the corresponding tokenizer.

        .. tip::
            If your tokenizer cannot be initialized by the default implementation,
            we recommend you to intialize and pass the tokenizer manually;
            or you can override this function (**It accepts no arguments**)
        """
        if self.model_name_or_path is not None:
            try:
                return self.tokenizer_class.from_pretrained(self.model_name_or_path)
            except:
                raise EnvironmentError(
                    f'Please make sure "{self.model_name_or_path}" is a repo or path where the tokenizer is saved')
        else:
            return None

    def prepare_weight(self, weight: np.array, name: str):
        """
        This function wraps a weight passed for initializing a module parameter and saves its `shape` and `dtype` as a
        in `BaseConfig.WEIGHT_DIMENSIONS`.
        So when :func:`~from_pretrained` is called without passing the weight (i.e. when `weight` is None),
        the corresponding parameter is initialized from the saved `shape` and `dtype`
        before :func:`~load_state_dict` is called to copy the module's weights.

        Args:
            weight: the weight that will be copied to a parameter of the module
            name: a name to distinguish the weight in the module config

        Returns:
            a pytorch tensor with the same value as `weight` if `weight` is not None, otherwise a zero tensor with
            the saved `shape` and `dtype` in the module config.
        """
        if weight is None:
            return torch.zeros(self.config.WEIGHT_DIMENSIONS[name + ".shape"],
                               dtype=eval(self.config.WEIGHT_DIMENSIONS[name + ".dtype"]))
        else:
            weight = torch.as_tensor(weight)
            self.config.WEIGHT_DIMENSIONS[name + ".shape"] = list(weight.shape)
            self.config.WEIGHT_DIMENSIONS[name + ".dtype"] = str(weight.dtype)
            return weight

    @monitor
    def response(self, raw_input, tokenizer, return_dict=False, **kwargs):
        r"""
        The main function for the module to generate a response given an input.

        .. note::
            Please refer to our tutorial for implementation guidance: :doc:`/development/overview`

        Args:
            raw_input (str): the text input
            tokenizer (PreTrainedTokenizer): the tokenizer used to tokenize the input
            return_dict (bool): if set to True, will return a dict of outputs instead of a single output
            **kwargs: the keyword arguments that will be passed to :func:`~forward`

        Returns:
            By default, a single output will be returned. If `return_dict` is set to True, a dict of outputs will be returned.
        """
        tokenized_input = tokenizer(raw_input, return_tensors="pt")
        logits = self.forward(tokenized_input, **kwargs)
        output = tokenizer.batch_decode(logits, skip_special_tokens=True)
        if return_dict:
            return {
                "input": raw_input,
                "tokenized_input": tokenized_input,
                "logits": logits,
                "output": output
            }
        else:
            return output

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        r"""

        .. note::
            Adapted from pytorch's original implementation to support :attr:`self.LOAD_SAVE_IGNORES`,
            when a module matches any pattern in :attr:`self.LOAD_SAVE_IGNORES`, it will be ignored in the returned state dict.

        --------------------------------

        Returns a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if name in self.LOAD_SAVE_IGNORES:
                logging.debug(f"Ignored state dict for {prefix + name}")
                continue
            module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)

        # remove chained destination
        destination = self.remove_ignores(self.LOAD_SAVE_IGNORES, destination)

        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def load_state_dict(self,
                        state_dict: 'OrderedDict[str, Tensor]',
                        strict: bool = True,
                        allow_unexpected=False,
                        LOAD_PREFIX: Optional[str] = '',
                        LOAD_MAPPINGS: Optional[dict] = None,
                        LOAD_IGNORES: Optional[Iterable[str]] = tuple(),
                        ):
        r"""
        .. note::
            Adapted from pytorch's original implementation to support partial loading and state dict mapping.

        --------------------------------

        Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        Args:
            allow_unexpected: if set to True, will not complain about Unexpected key(s)
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            LOAD_PREFIX: When set, will load state dict from a prefix `LOAD_PREFIX`
            LOAD_MAPPINGS: When set, for each `key`, `val` pair in the dict, any parameter in state_dict
                that has `key` as a prefix will be mapped to one that has `val` in place of `key` as a new prefix.
                (This parameter is useful when the submodules/parameters have different names from their counterparts
                in the state dict.)
            LOAD_IGNORES: When set, the submodules in the state_dict that match any pattern in `LOAD_IGNORES`
                        will be ignored in loading. Note that `self.LOAD_SAVE_IGNORES` will be merged into `LOAD_IGNORES`.
                        (This option can be useful when you use a frozen pre-trained model inside your model
                        to avoid duplicate loading and saving)

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """

        LOAD_IGNORES = self.LOAD_SAVE_IGNORES | set(LOAD_IGNORES)
        if LOAD_MAPPINGS is None:
            LOAD_MAPPINGS = dict()
        state_dict = self.map_parameters(LOAD_MAPPINGS, state_dict)
        state_dict = self.remove_ignores(LOAD_IGNORES, state_dict)

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, prefix=''):
            if prefix.strip('.') in LOAD_IGNORES:  # skip chained LOAD_IGNORE. e.g. processor.encoder
                return
            if module.__dict__.get(
                    "LOAD_IGNORES") and "" in module.LOAD_IGNORES:  # skip the whole module when specified
                return
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if module.__dict__.get(
                        "LOAD_IGNORES") and name in module.LOAD_IGNORES:  # skip submodules when specified
                    continue
                if child is not None:
                    load(child, prefix + name + '.')
            return

        load(self, prefix=LOAD_PREFIX)
        del load

        if not allow_unexpected and len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            if strict:
                raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, "\n\t".join(error_msgs)))
            else:  # NOTE: I modified the original implementation for debug
                for msg in error_msgs:
                    logging.debug(msg)
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    @staticmethod
    def map_parameters(name_mappings, state_dict, remove_origin=True):
        """
        Map parameters in state_dict to another name according to ``name_mappings``.

        Args:
            name_mappings (dict): a dict that maps the original parameter name to the new name.
            state_dict (dict): the state dict to be mapped.
            remove_origin (bool): whether to remove the original parameter.

        Returns:
            dict: the mapped state dict.

        Example:
            >>> state_dict = {'block1.weight': tensor1, 'block1.bias': tensor2, 'block2.weight': tensor3}
            >>> name_mappings = {'block1.': 'module1.', 'block2.': 'module2.'}
            >>> mapped_state_dict = module.map_parameters(name_mappings, state_dict, remove_origin=False)
            {'module1.weight': tensor1, 'module1.bias': tensor2, 'module2.weight': tensor3, 'block1.weight': tensor1, 'block1.bias': tensor2, 'block2.weight': tensor3}

        """
        for key, value in list(state_dict.items()):
            for prefix, mapped_prefix in name_mappings.items():
                if key.startswith(prefix) and prefix != mapped_prefix:
                    new_key = mapped_prefix + key[len(prefix):]
                    state_dict[new_key] = copy.deepcopy(value)
                    if remove_origin:
                        state_dict.pop(key)
                    break
        return state_dict

    @staticmethod
    def remove_ignores(load_ignores: Iterable, state_dict):
        """
        Remove parameters in state_dict according to ``load_ignores``.

        Args:
            load_ignores (list): a list of prefixes to be ignored.
            state_dict (dict): the state dict to operate on.

        Returns:
            dict: the state dict after removing ignored parameters.
        """
        used_prefix = set()
        for key in list(state_dict.keys()):
            for prefix in load_ignores:
                if key.startswith(prefix):
                    used_prefix.add(prefix)
                    state_dict.pop(key)
                    break

        for prefix in used_prefix:
            logging.debug(f"Ignored state dict for {prefix}")
        return state_dict

    def load_checkpoint(self, checkpoint, verbose=True, strict=True, **kwargs):
        """
        Load a checkpoint from a file.

        Args:
            checkpoint (str): the path to the checkpoint file.
            verbose (bool): whether to print the message when loading the checkpoint.
            strict (bool): the strict argument passed to `load_state_dict`.
        """
        state_dict = torch.load(checkpoint, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.load_state_dict(state_dict, strict, **kwargs)
        if verbose:
            logging.info(f"{self.__class__.__name__} loaded checkpoint '{checkpoint}'")

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            **kwargs,
    ):
        """
        Saves the `pretrained_model_name_or_path` as a member variable of the model, so when
        `get_tokenizer` is called, the tokenizer can be initialized with `model_name_or_path`

        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.model_name_or_path = pretrained_model_name_or_path
        return model

# def bind_weights(module1: torch.nn.Module, module2: torch.nn.Module):
#     """
#     Bind the weights of two PyTorch modules so they share the same memory space.
#
#     Args:
#         module1 (torch.nn.Module): the first module.
#         module2 (torch.nn.Module): the second module.
#     """
#     for param1, param2 in zip(module1.parameters(), module2.parameters()):
#         param2.data.set_(param1.data)
#         param2.requires_grad = param1.requires_grad
