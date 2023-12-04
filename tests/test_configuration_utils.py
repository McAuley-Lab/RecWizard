from recwizard.configuration_utils import BaseConfig

def test_base_config_init():
    # Example values for testing
    WEIGHT_DIMENSIONS = {"param1": {"dtype": "float32", "shape": (256,)}}
    additional_kwargs = {"num_layers": 12, "hidden_size": 768}

    config = BaseConfig(WEIGHT_DIMENSIONS=WEIGHT_DIMENSIONS, **additional_kwargs)

    # Check if the inherited parameters are correctly set
    assert config.num_layers == 12
    assert config.hidden_size == 768

    # Check if the custom parameter is correctly set
    assert config.WEIGHT_DIMENSIONS == WEIGHT_DIMENSIONS

def test_base_config_default_values():
    config = BaseConfig()

    # Check if default values are correctly set
    assert config.WEIGHT_DIMENSIONS == {}