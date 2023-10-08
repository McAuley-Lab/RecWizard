from typing import List

from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utility.utils import WrapSingleInput

class KBRDGenTokenizer(BaseTokenizer):

    @WrapSingleInput
    def decode(
            self,
            ids,
            *args,
            **kwargs,
    ) -> List[str]:
        return self.tokenizers[0].decode(ids)
    
    def __call__(self, *args, **kwargs):
        kwargs.update(return_tensors='pt', padding=True, truncation=True)
        return super().__call__(*args, **kwargs)
