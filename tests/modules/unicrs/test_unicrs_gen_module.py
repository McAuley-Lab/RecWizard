import pytest
from recwizard.modules.unicrs import UnicrsGen

class TestUnicrsGenModule:
    def setup_method(self):
        self.model = UnicrsGen.from_pretrained("recwizard/unicrs-gen")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None
        