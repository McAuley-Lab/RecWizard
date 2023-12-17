import pytest
from recwizard.modules.kgsf import KGSFGen

class TestKGSFGenModule:
    def setup_method(self):
        self.model = KGSFGen.from_pretrained("recwizard/kgsf-gen")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None