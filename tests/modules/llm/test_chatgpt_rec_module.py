from recwizard.modules.zero_shot import ChatgptRec


class TestChatGPTRecModule:
    def setup_method(self):
        import os

        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
        self.model = ChatgptRec.from_pretrained("recwizard/chatgpt-rec-fillblank")
        self.tokenizer = self.model.get_tokenizer()

    def test_response(self):
        pass
