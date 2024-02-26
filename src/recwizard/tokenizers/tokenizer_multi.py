""" In this module, we define the MultiTokenizer class which is a wrapper around multiple tokenizers.

    The MultiTokenizer class is a wrapper around multiple tokenizers. It is used to combine multiple tokenizers into a single
    tokenizer. It is used to encode and decode text using multiple tokenizers. It is used to save and load multiple tokenizers
    together.
"""

import os, warnings
import json

from typing import Union, Tuple, Any

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TOKENIZER_CONFIG_FILE
from transformers.tokenization_utils import BatchEncoding
from transformers import AutoTokenizer

from huggingface_hub import hf_hub_download

from collections import OrderedDict

from ..modules.kgsf import KGSFBaseTokenizer
from recwizard.tokenizers import NLTKTokenizer, EntityTokenizer

REGISTERED_TOKENIZERS = {
    "NLTKTokenizer": NLTKTokenizer,
    "EntityTokenizer": EntityTokenizer,
    "KGSFBaseTokenizer": KGSFBaseTokenizer,
}


class MultiTokenizer(PreTrainedTokenizerBase):

    def __init__(self, tokenizers: Union[dict, OrderedDict] = None, tokenizer_key_for_decoding: str = None, **kwargs):
        # Set the keys as ordered for the tokenizers
        self.tokenizers = OrderedDict(tokenizers)
        self.tokenizer_key_for_decoding = tokenizer_key_for_decoding
        self.tokenizer_for_decoding = self.tokenizers[tokenizer_key_for_decoding]

        super().__init__(**kwargs)

    def __call__(self, text, *args: Any, **kwargs: Any) -> Any:
        if type(text) == dict:
            encodings = {}
            for key in text:
                assert key in self.tokenizers, f"Tokenizer for {key} not found in the tokenizers"
                encodings[key] = self.tokenizers[key](text[key], *args, **kwargs)
            return BatchEncoding(encodings)
        else:
            encodings = {key: tokenizer(text, *args, **kwargs) for key, tokenizer in self.tokenizers.items()}
        return BatchEncoding(encodings)

    def encode(self, text: str, *args, **kwargs):
        if type(text) == dict:
            encodings = {}
            for key in text:
                assert key in self.tokenizers, f"Tokenizer for {key} not found in the tokenizers"
                encodings[key] = self.tokenizers[key].encode(text[key], *args, **kwargs)
            return BatchEncoding(encodings)
        else:
            encodings = {key: tokenizer.encode(text, *args, **kwargs) for key, tokenizer in self.tokenizers.items()}
        return BatchEncoding(encodings)

    def decode(self, tokens: Union[list, int], *args, **kwargs):
        return self.tokenizer_for_decoding.decode(tokens, *args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer_for_decoding.batch_decode(*args, **kwargs)

    def apply_chat_template(self, text, *args, **kwargs):
        return {key: tokenizer.apply_chat_template(text, *args, **kwargs) for key, tokenizer in self.tokenizers.items()}

    def __len__(self):
        return len(self.tokenizers)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        """
        Save the full tokenizer state.


        This method make sure the full tokenizer can then be re-loaded using the
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] class method..

        Warning,None This won't save modifications you may have applied to the tokenizer after the instantiation (for
        instance, modifying `tokenizer.do_lower_case` after creation).

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.

        Returns:
            A tuple of `str`: The files saved.
        """

        # Check auth token and issue warning if needed
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        # Check if the provided path is a directory
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        # Create the folder
        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Save config of the multiple tokenizer
        config_file_path = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)
        config = {
            "tokenizers": {key: tokenizer.__class__.__name__ for key, tokenizer in self.tokenizers.items()},
            "tokenizer_key_for_decoding": self.tokenizer_key_for_decoding,
        }
        with open(config_file_path, "w") as json_file:
            json.dump(config, json_file, indent=2)

        # Save the tokenizers in this directory
        for key, tokenizer in self.tokenizers.items():
            tokenizer.save_pretrained(os.path.join(save_directory, key))

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=kwargs.pop("commit_message", None),
                token=kwargs.get("token"),
            )

        return ["multi_tokenizer_config.json"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the tokenizer from pretrained model or local directory.
        It loads the initialization kwargs from the 'tokenizer_kwargs.json' file before initializing the tokenizer.

        Args:
            pretrained_model_name_or_path (str): The path to the pretrained model or the name of the pretrained model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            MultiTokenizer: The tokenizer loaded from the pretrained model or local directory.
        """
        try:
            path = hf_hub_download(pretrained_model_name_or_path)
        except:
            path = os.path.join(pretrained_model_name_or_path)

        # Load the config file
        config_file = os.path.join(path, TOKENIZER_CONFIG_FILE)
        with open(config_file, "r") as json_file:
            config = json.load(json_file)

        # Load the tokenizers
        tokenizers = {}
        for t in config["tokenizers"]:
            if config["tokenizers"][t] in REGISTERED_TOKENIZERS:
                tokenizer_class = REGISTERED_TOKENIZERS[config["tokenizers"][t]]
            else:
                tokenizer_class = AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(os.path.join(path, t))
            tokenizers[t] = tokenizer

        return cls(
            tokenizers=tokenizers, tokenizer_key_for_decoding=config["tokenizer_key_for_decoding"], *args, **kwargs
        )
