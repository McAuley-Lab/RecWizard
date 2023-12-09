import pytest
from recwizard.modules.unicrs import UnicrsRec

class TestUnicrsRecModule:
    def setup_method(self):
        self.model = UnicrsRec.from_pretrained("recwizard/unicrs-rec")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None