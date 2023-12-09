import pytest
from recwizard.modules.redial import RedialGen

class TestRedialGenModule:
    def setup_method(self):
        self.model = RedialGen.from_pretrained("recwizard/redial-gen")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None