from recwizard.configuration_utils import BaseConfig

class KBRDGenConfig(BaseConfig):
    """
    The configuration for the generator in the KBRD model.

    Args:
        gen_dim (int): Dimensionality of the generator component.
        rec_dim (int): Dimensionality of the recommender component.
        vocab_size (int): Size of the vocabulary used in the model.
        pad_idx (int): Index representing padding in the vocabulary.
        start_idx (int): Index representing the start token in the vocabulary.
        end_idx (int): Index representing the end token in the vocabulary.
        n_positions (int): Number of positions in positional embeddings.
        n_heads (int): Number of attention heads in the model.
        n_layers (int): Number of transformer layers in the model.
        ffn_size (int): Size of the feedforward network in each layer.
        dropout (float): Dropout rate applied to various layers.
        attention_dropout (float): Dropout rate applied to attention layers.
        relu_dropout (float): Dropout rate applied to ReLU activation layers.
        learn_positional_embeddings (bool): Whether to learn positional embeddings.
        embeddings_scale (bool): Whether to scale embeddings at initialization.
        rec_module_config (dict): Configuration settings for the recommender module.
    """
    def __init__(self,
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
                 **kwargs):
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
        self.rec_module_config = rec_module_config # include another config

