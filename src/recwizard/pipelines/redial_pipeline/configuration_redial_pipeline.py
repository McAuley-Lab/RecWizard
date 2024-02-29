from recwizard.configuration_utils import BaseConfig


class RedialPipelineConfig(BaseConfig):
    """
    The configuration of the expansion pipeline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
