# Basic information
__version__ = "0.1.0"


# Monitor for UI
from .monitor_utils import monitoring, RecwizardMonitor

monitor = RecwizardMonitor.monitor


# Base classes
from .configuration_utils import BaseConfig
from .module_utils import BaseModule
from .pipeline_utils import BasePipeline


# Implemented pipelines
from .pipelines.expansion import ExpansionPipeline, ExpansionConfig
from .pipelines.fill_blank import FillBlankPipeline, FillBlankConfig
from .pipelines.redial_only import RedialOnlyPipeline, RedialOnlyConfig
from .pipelines.trivial import TrivialPipeline, TrivialConfig
from .pipelines.chatgpt_merge import ChatGPTMergePipeline, ChatGPTMergeConfig


# Implemented modules
from .modules.redial import (
    RedialGen,
    RedialRec,
    RedialGenConfig,
    RedialRecConfig,
    RedialGenTokenizer,
    RedialRecTokenizer,
)
from .modules.unicrs import (
    UnicrsGen,
    UnicrsRec,
    UnicrsGenConfig,
    UnicrsRecConfig,
    UnicrsGenTokenizer,
    UnicrsRecTokenizer,
)
from .modules.kbrd import KBRDGen, KBRDRec, KBRDGenConfig, KBRDRecConfig, KBRDGenTokenizer, KBRDRecTokenizer
from .modules.zero_shot import (
    HFLLMGen,
    HFLLMGenConfig,
    ChatGPTGen,
    ChatGPTRec,
    ChatGPTGenConfig,
    ChatGPTRecConfig,
    ChatGPTTokenizer,
)
