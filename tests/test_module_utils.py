import pytest
from recwizard.module_utils import BaseModule



class TestBaseModule:
    def setup_method(self):
        self.model = BaseModule.from_pretrained("hf-internal-testing/tiny-bert")
        self.tokenizer = self.model.get_tokenizer()
    
    def test_get_tokenizer(self):    
        assert self.tokenizer is not None
    
    def test_from_pretrained_module(self):
        assert self.model.model_name_or_path == "hf-internal-testing/tiny-bert"

    def test_map_parameters(self):
        parameters = {"a": 1, "b": 2, "c": 3}
        mapping = {"a": "b", "b": "c", "c": "a"}
        mapped_parameters = BaseModule.map_parameters(mapping, parameters)
        assert mapped_parameters == {"b": 1, "c": 2, "a": 3}

    def test_remove_ignores(self):
        parameters = {"a": 1, "a.b": 2, "c": 3}
        ignores = {"a"}
        parameters = BaseModule.remove_ignores(ignores, parameters)
        assert parameters == {"c": 3}

    
