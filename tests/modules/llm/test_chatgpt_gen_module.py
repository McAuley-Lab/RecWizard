from recwizard.modules.llm import ChatgptGen

class TestChatGPTGenModule:
    def setup_method(self):
        import os
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
        self.model = ChatgptGen.from_pretrained("recwizard/chatgpt-gen-expansion")
        self.tokenizer = self.model.get_tokenizer()
    def test_response(self):
        # we don't test chatgpt api
        pass