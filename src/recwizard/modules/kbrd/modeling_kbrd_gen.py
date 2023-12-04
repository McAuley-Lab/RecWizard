from typing import Union, List
from transformers.utils import ModelOutput

import torch

from recwizard.module_utils import BaseModule
from .configuration_kbrd_gen import KBRDGenConfig
from .tokenizer_kbrd_gen import KBRDGenTokenizer
from .modeling_kbrd_rec import KBRDRecConfig, KBRDRec
from .transformer_encoder_decoder import TransformerGeneratorModel

from recwizard.modules.monitor import monitor

class KBRDGen(BaseModule):
    """The generator component of the KBRD model.

    Args:
        config (KBRDGenConfig): Configuration settings for the generator.
        **kwargs: Additional keyword arguments.

    Attributes:
        model (TransformerGeneratorModel): The underlying Transformer-based generator model.
    """
    config_class = KBRDGenConfig
    tokenizer_class = KBRDGenTokenizer
    LOAD_SAVE_IGNORES = {'encoder.text_encoder', 'decoder'}

    def __init__(self, config: KBRDGenConfig, **kwargs):

        super().__init__(config, **kwargs)
        
        self.model = TransformerGeneratorModel(
            config, 
            kbrd_rec=KBRDRec(
                config=KBRDRecConfig.from_dict(config.rec_module_config)
            )
        )
        
    def generate(self, 
                 input_ids: torch.LongTensor = None, 
                 labels: torch.LongTensor = None, 
                 entities: torch.LongTensor = None,
                 entity_attention_mask: torch.BoolTensor = None,
                 bsz: int = None,
                 maxlen: int = 64):
        """Generate responses based on input data.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            labels (torch.LongTensor): Target token IDs for training.
            entities (torch.LongTensor): Entity IDs.
            entity_attention_mask (torch.BoolTensor): Entity attention mask.
            bsz (int): Batch size.
            maxlen (int): Maximum length for generated responses.

        Returns:
            tuple: A tuple containing:
                - torch.FloatTensor: Logits.
                - torch.LongTensor: Predicted token IDs.
                - List[str]: Decoded output responses.
        """
        if bsz is None:
            bsz = input_ids.size(0) 

        if entities is not None:
            user_representation, _ = self.model.kbrd.model.user_representation(entities, entity_attention_mask)
            self.model.user_representation = user_representation.detach()

        return self.model(*(input_ids,), bsz=bsz, ys=labels, maxlen=maxlen)
    
    def response(self, 
                 raw_input: Union[List[str], str], 
                 tokenizer: KBRDGenTokenizer, 
                 return_dict=False):
        """Generate responses given raw input data.

        Args:
            raw_input (Union[List[str], str]): Raw input text or list of text.
            tokenizer (KBRDGenTokenizer): Tokenizer for processing input data.
            return_dict (bool, optional): Whether to return results as a dictionary.

        Returns:
            Union[List[str], dict]: Generated responses as a list of strings or a dictionary with additional information.
        """
        input_ids = torch.LongTensor(tokenizer(raw_input)['input_ids']).to(self.device)
        entities = torch.LongTensor(tokenizer.tokenizers[-1](raw_input)['entities']).to(self.device).unsqueeze(dim=0)
        inputs = {
            'input_ids': input_ids,
            'entities': entities,
            'entity_attention_mask': entities != tokenizer.tokenizers[-1].pad_entity_id,
        }
        outputs = self.generate(**inputs)

        # mask all non-movie indices
        if return_dict:
            return {
                'logits': outputs[0],
                'pred_ids': outputs[1],
                'output': tokenizer.batch_decode(outputs[1].tolist())
            }

        return tokenizer.batch_decode(outputs[1].tolist())
