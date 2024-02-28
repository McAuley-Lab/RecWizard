from recwizard.modules.zero_shot import LlamaGen


class TestLlamaGenModule:
    def setup_method(self):
        self.model = LlamaGen.from_pretrained("recwizard/llama-expansion").to("cuda")
        self.tokenizer = self.model.get_tokenizer()

    def test_response(self):
        assert self.model.response("hello", self.tokenizer) is not None
