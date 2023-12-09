import pytest

from recwizard.modules.redial import RedialRec

class TestRedialRecModule:
    def setup_method(self):
        self.model = RedialRec.from_pretrained("recwizard/redial-rec")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None
