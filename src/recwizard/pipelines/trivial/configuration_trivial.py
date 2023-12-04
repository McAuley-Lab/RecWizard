from recwizard.configuration_utils import BaseConfig

class TrivialConfig(BaseConfig):
    """
    Configuration class for a trivial model type.
    """
    model_type = "trivial"

    def __init__(self, **kwargs):
        """
        Initialize the TrivialConfig.

        Args:
            **kwargs: Additional keyword arguments to be passed to the BaseConfig constructor.
        """
        super().__init__(**kwargs)
