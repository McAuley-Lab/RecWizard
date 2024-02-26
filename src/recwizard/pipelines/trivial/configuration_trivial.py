from recwizard.configuration_utils import BaseConfig


class TrivialConfig(BaseConfig):
    model_type = "trivial"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
