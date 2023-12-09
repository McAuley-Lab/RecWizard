from recwizard.configuration_utils import BaseConfig


class KBRDGenConfig(BaseConfig):
    """Configuration class to store the configuration of a `KBRDGen`."""

    def __init__(
        self,
        gen_dim: int = None,
        rec_dim: int = None,
        vocab_size: int = None,
        pad_idx: int = None,
        start_idx: int = None,
        end_idx: int = None,
        n_positions: int = None,
        n_heads: int = None,
        n_layers: int = None,
        ffn_size: int = None,
        dropout: float = 0,
        attention_dropout: float = 0,
        relu_dropout: float = 0,
        learn_positional_embeddings: bool = False,
        embeddings_scale: bool = True,
        rec_module_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gen_dim = gen_dim
        self.rec_dim = rec_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.n_positions = n_positions
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.learn_positional_embeddings = learn_positional_embeddings
        self.embeddings_scale = embeddings_scale
        self.rec_module_config = rec_module_config  # include another config
