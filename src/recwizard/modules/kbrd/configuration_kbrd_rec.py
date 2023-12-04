from recwizard.configuration_utils import BaseConfig

class KBRDRecConfig(BaseConfig):
    """
    Configuration class for the KBRD (Knowledge-based Recommendation) model.

    Args:
        n_entity (int): Number of entities in the knowledge graph.
        n_relation (int): Number of relations in the knowledge graph.
        sub_n_relation (int): Number of sub-relations used in the model.
        dim (int): Dimensionality of the model's embeddings.
        num_bases (int): Number of bases for the factorization in the relation modeling.
    """
    def __init__(self,
                 n_entity: int = None,
                 n_relation: int = None,
                 sub_n_relation: int = None,
                 dim: int = None,
                 num_bases: int = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.sub_n_relation = sub_n_relation
        self.dim = dim
        self.num_bases = num_bases
