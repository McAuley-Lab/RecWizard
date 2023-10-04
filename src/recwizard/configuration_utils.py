from transformers import PretrainedConfig

class BaseConfig(PretrainedConfig):
    r"""
    The base config for all modules/pipelines.
    """
    def __init__(self, WEIGHT_DIMENSIONS=None, **kwargs):
        """

        Args:
            WEIGHT_DIMENSIONS (dict, optional): The dimension and dtype of module parameters.
                Used to initialize the parameters when they are not explicitly specified in module initialization.
                Defaults to None. See also :func:`recwizard.module_utils.BaseModule.prepare_weight`.
            **kwargs: Additional parameters. Will be passed to the `PretrainedConfig.__init__`.

        """
        super().__init__(**kwargs)
        self.WEIGHT_DIMENSIONS = {} if WEIGHT_DIMENSIONS is None else WEIGHT_DIMENSIONS

