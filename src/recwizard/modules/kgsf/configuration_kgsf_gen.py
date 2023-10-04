from recwizard.configuration_utils import BaseConfig


class KGSFGenConfig(BaseConfig):
    def __init__(self, 
                 batch_size: int, 
                 max_r_length: int, 
                 embedding_size: int, 
                 n_concept: int, 
                 dim: int, 
                 n_entity: int, 
                 num_bases: int, 
                 n_positions: int = None,
                 truncate: int = 0, 
                 text_truncate: int = 0, 
                 label_truncate: int = 0, 
                 padding_idx: int = 0, 
                 start_idx: int = 1, 
                 end_idx: int = 2, 
                 longest_label: int = 1, 
                 pretrain: bool = False,
                 n_heads: int = 2,
                 n_layers: int = 2,
                 ffn_size: int = 300,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.0,
                 relu_dropout: float = 0.1,
                 learn_positional_embeddings: bool = False,
                 embeddings_scale: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.max_r_length = max_r_length
        self.embedding_size = embedding_size
        self.n_concept = n_concept
        self.dim = dim
        self.n_entity = n_entity

        if n_positions == None:  # should default to 1024 if truncate,text_truncate,label_truncate are 0
            if max(truncate,label_truncate,label_truncate) != 0:
                self.n_positions = max(truncate,label_truncate,label_truncate)
            else:
                self.n_positions = 1024
        else:
            self.n_positions = n_positions

        self.num_bases = num_bases
        self.truncate = truncate
        self.text_truncate = text_truncate
        self.label_truncate = label_truncate
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.longest_label = longest_label
        self.pretrain = pretrain
        
        # for building encoder
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_size = ffn_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.relu_dropout = relu_dropout
        self.learn_positional_embeddings = learn_positional_embeddings
        self.embeddings_scale = embeddings_scale