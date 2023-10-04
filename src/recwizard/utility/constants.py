STATE_DICT_FILE = "pytorch_model.bin"
"""
The default name of the state dict file
"""

BOS_TOKEN: str = "<bos>"
SEP_TOKEN: str = "<sep>"
EOS_TOKEN: str = "<eos>"

HF_ORG = "recwizard"
"""
Our official organization that's used to publish new models and tokenizers
"""

ENTITY = 'entity'
ENTITY_PATTERN = r'<entity>(.*?)</entity>'
ENTITY_TEMPLATE = "<entity>{}</entity>"
