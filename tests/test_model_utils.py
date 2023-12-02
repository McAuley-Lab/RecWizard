import pytest
from transformers import PretrainedConfig
from recwizard.configuration_utils import BaseConfig
from recwizard.module_utils import BaseModule
from recwizard import BasePipeline

# Mock classes for testing
class MockPretrainedConfig(PretrainedConfig):
    pass

class MockBaseConfig(MockPretrainedConfig, BaseConfig):
    pass

class MockBaseModule(BaseModule):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.LOAD_SAVE_IGNORES = set()

    def get_tokenizer(self):
        return MockBaseTokenizer()

class MockBaseTokenizer:
    pass

# Tests
def test_base_pipeline_init():
    rec_module = MockBaseModule(config=MockBaseConfig())
    gen_module = MockBaseModule(config=MockBaseConfig())
    rec_tokenizer = MockBaseTokenizer()
    gen_tokenizer = MockBaseTokenizer()

    pipeline = BasePipeline(
        config=MockBaseConfig(),
        rec_module=rec_module,
        gen_module=gen_module,
        rec_tokenizer=rec_tokenizer,
        gen_tokenizer=gen_tokenizer
    )

    assert pipeline.rec_module is rec_module
    assert pipeline.gen_module is gen_module
    assert pipeline.rec_tokenizer is rec_tokenizer
    assert pipeline.gen_tokenizer is gen_tokenizer
    assert pipeline.config_class == BaseConfig

def test_base_pipeline_response(mocker):
    rec_module = MockBaseModule(config=MockBaseConfig())
    gen_module = MockBaseModule(config=MockBaseConfig())
    rec_tokenizer = MockBaseTokenizer()
    gen_tokenizer = MockBaseTokenizer()

    pipeline = BasePipeline(
        config=MockBaseConfig(),
        rec_module=rec_module,
        gen_module=gen_module,
        rec_tokenizer=rec_tokenizer,
        gen_tokenizer=gen_tokenizer
    )

    mocker.patch.object(rec_module, 'response', return_value='rec_output')
    mocker.patch.object(gen_module, 'response', return_value='gen_output')

    query = "Test query"
    result = pipeline.response(query, return_dict=True)

    assert result == {'rec_output': 'rec_output', 'gen_output': 'gen_output'}

def test_base_pipeline_response_with_args(mocker):
    rec_module = MockBaseModule(config=MockBaseConfig())
    gen_module = MockBaseModule(config=MockBaseConfig())
    rec_tokenizer = MockBaseTokenizer()
    gen_tokenizer = MockBaseTokenizer()

    pipeline = BasePipeline(
        config=MockBaseConfig(),
        rec_module=rec_module,
        gen_module=gen_module,
        rec_tokenizer=rec_tokenizer,
        gen_tokenizer=gen_tokenizer
    )

    mocker.patch.object(rec_module, 'response', return_value='rec_output')
    mocker.patch.object(gen_module, 'response', return_value='gen_output')

    query = "Test query"
    rec_args = {'arg1': 'value1'}
    gen_args = {'arg2': 'value2'}
    result = pipeline.response(query, return_dict=True, rec_args=rec_args, gen_args=gen_args)

    assert result == {'rec_output': 'rec_output', 'gen_output': 'gen_output'}

def test_base_pipeline_response_without_return_dict():
    rec_module = MockBaseModule(config=MockBaseConfig())
    gen_module = MockBaseModule(config=MockBaseConfig())
    rec_tokenizer = MockBaseTokenizer()
    gen_tokenizer = MockBaseTokenizer()

    pipeline = BasePipeline(
        config=MockBaseConfig(),
        rec_module=rec_module,
        gen_module=gen_module,
        rec_tokenizer=rec_tokenizer,
        gen_tokenizer=gen_tokenizer
    )

    query = "Test query"
    with pytest.raises(NotImplementedError):
        pipeline.response(query, return_dict=False)

def test_base_pipeline_forward_not_implemented():
    rec_module = MockBaseModule(config=MockBaseConfig())
    gen_module = MockBaseModule(config=MockBaseConfig())
    rec_tokenizer = MockBaseTokenizer()
    gen_tokenizer = MockBaseTokenizer()
    pipeline = BasePipeline(
        config=MockBaseConfig(),
        rec_module=rec_module,
        gen_module=gen_module,
        rec_tokenizer=rec_tokenizer,
        gen_tokenizer=gen_tokenizer
    )
    with pytest.raises(NotImplementedError):
        pipeline.forward()
