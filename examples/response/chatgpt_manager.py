"""
Test result of this code:

('User: Hi. I like everything but fantasy films and alien type stuff. have you seen anything good lately?\n', "System: I haven't seen The Lord of the Rings yet. I'm not sure what I can find.")
"""

import sys
from recwizard.modules.unicrs import UnicrsGen, UnicrsRec
from recwizard.pipelines.chatgpt import ChatgptAgentConfig, ChatgptAgent

if __name__ == '__main__':
    model = ChatgptAgent(
        config=ChatgptAgentConfig(),
        gen_module=UnicrsGen.from_pretrained('recwizard/unicrs-gen-redial'),
        rec_module=UnicrsRec.from_pretrained('recwizard/unicrs-rec-redial'),
    )
    query = 'User: Hi. I like everything but <entity>fantasy</entity> films and alien type stuff.' \
                ' have you seen anything good lately?'

    print(model.response(query))
