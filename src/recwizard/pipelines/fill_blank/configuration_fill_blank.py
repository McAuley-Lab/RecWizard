from recwizard.configuration_utils import BaseConfig


class FillBlankConfig(BaseConfig):
    """The configuration of the fill blank pipeline.

    Attributes:
        rec_pattern (str): The pattern to be used as the item placeholders for filling-blank pipeline.
    """

    def __init__(self, rec_pattern: str = r"<movie>", **kwargs):
        """Initialize the configuration.

        Args:
            rec_pattern (str, optional): The pattern to be used as the item placeholders for filling-blank pipeline. Defaults to r"<movie>".
        """
        super().__init__(**kwargs)
        self.rec_pattern = rec_pattern
