from recwizard.configuration_utils import BaseConfig


class KBRDRecConfig(BaseConfig):
    """Configuration class to store the configuration of a `KBRDRec`."""

    def __init__(
        self,
        n_entity: int = None,
        n_relation: int = None,
        sub_n_relation: int = None,
        dim: int = None,
        num_bases: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.sub_n_relation = sub_n_relation
        self.dim = dim
        self.num_bases = num_bases
