STATE_DICT_FILE = "pytorch_model.bin"
"""
The default name of the state dict file
"""

BOS_TOKEN: str = "<bos>"
SEP_TOKEN: str = "<sep>"
EOS_TOKEN: str = "<eos>"

START_TAG = "<entity>"
END_TAG = "</entity>"

HF_ORG = "recwizard"
"""
Our official organization that's used to publish new models and tokenizers
"""

ENTITY = "entity"
ENTITY_PATTERN = r"<entity>(.*?)</entity>"
ENTITY_TEMPLATE = "<entity>{}</entity>"

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{% if not loop.last %}{{'\n'}}{% endif %}{% endfor %}"

ASSISTANT_TOKEN = "System:"
USER_TOKEN = "User:"

from .entity_linkers import EntityLink
from .decorators import Singleton, WrapSingleInput
from .chat_template import create_chat_message, create_item_list
from .others import *
from .device_manager import DeviceManager
