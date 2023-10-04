from recwizard.configuration_utils import BaseConfig

class KBRDRecConfig(BaseConfig):
    def __init__(self,
                 n_entity: int = None,
                 n_relation: int = None,
                 dim: int = None,
                 kg_path: str = None,
                 num_bases: int = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.kg_path = kg_path
        self.num_bases = num_bases
