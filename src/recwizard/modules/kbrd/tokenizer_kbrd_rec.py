from typing import List

from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utility.utils import WrapSingleInput


class KBRDRecTokenizer(BaseTokenizer):
    @WrapSingleInput
    def decode(
        self,
        ids,
        *args,
        **kwargs,
    ) -> List[str]:
        """Decode a list of token ids into a list of strings.
        
        Args:
            ids (List[int]): list of token ids to decode;
        
        Returns:
            List[str]: list of decoded strings;
        """
        return [self.id2entity[id] for id in ids if id in self.id2entity]

    def __call__(self, *args, **kwargs):
        """Tokenize a string into a list of token ids."""
        kwargs.update(return_tensors="pt", padding=True, truncation=True)
        return super().__call__(*args, **kwargs)
