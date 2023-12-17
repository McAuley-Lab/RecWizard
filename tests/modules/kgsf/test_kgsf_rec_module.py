import pytest
from recwizard.modules.kgsf import KGSFRec

class TestKGSFRecModule:
    def setup_method(self):
        self.model = KGSFRec.from_pretrained("recwizard/kgsf-rec")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None
