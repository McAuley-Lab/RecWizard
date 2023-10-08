"""
Test result of this code:

('User: Hi. I like everything but fantasy films and alien type stuff. have you seen anything good lately? ', "System: Sure, if you're not into fantasy and alien films, you might enjoy The Lord of the Rings.")
"""

import sys

sys.path.append('./src')

from recwizard.modules.chatgpt import LLMConfig, LlamaGen
from recwizard.modules.unicrs import UnicrsRec
from recwizard.modules.redial import RedialRec
from recwizard import ExpansionConfig, ExpansionPipeline
import json

if __name__ == '__main__':

    # upload to hub if prompt is changed
    module = LlamaGen(LLMConfig())
    # module.push_to_hub('recwizard/llama-expansion')

    model = ExpansionPipeline(
        config=ExpansionConfig(),
        # gen_module=ChatgptGen.from_pretrained('recwizard/chatgpt-gen-expansion'),
        gen_module=module.to('cuda:0'),
        # rec_module=UnicrsRec.from_pretrained('recwizard/unicrs-rec-redial'),
        rec_module=RedialRec.from_pretrained('recwizard/redial-rec').to('cuda:0'),
        use_rec_logits=False
    ).to('cuda:0')

    query = ('User: Hi I am looking for movies like <entity>Super Troopers (2001)</entity>'
             # '<sep>System: You should watch <entity>Police Academy (1984)</entity>'
             # '<sep>User: Is that a great one? I have never seen it. I have seen <entity>American Pie</entity> I mean <entity>American Pie (1999)</entity>'
             # '<sep>System: Yes <entity>Police Academy (1984)</entity> is very funny and so is <entity>Police Academy 2: Their First Assignment (1985)</entity>'
             # '<sep>User: It sounds like I need to check them out'
             # '<sep>System: yes you will enjoy them'
             # '<sep>User: I appreciate your time. I will need to check those out. Are there any others you would recommend?'
             # '<sep>System: yes <entity>Lethal Weapon (1987)<entity>'
             # '<sep>User: Thank you i will watch that too'
             # '<sep>System: and also <entity>Beverly Hills Cop (1984)</entity>'
             # '<sep>User: Thanks for the suggestions.'
              # '<sep>System: you are welcome'
              )
    query1 = ('User: Hello!'
              '<sep>System: Hello, I have some movie ideas for you. Have you watched the movie <entity>Forever My Girl (2018)</entity> ?'
              '<sep>User: Looking for movies in the comedy category. I like Adam Sandler movies like <entity>Billy Madison (1995)</entity> Oh no is that good?')
    query2 = "User: Hi can you recommend a few <entity>fantasy</entity> movies to me?"
    print(model.response(query2, rec_args={'topk': 10}, gen_args={'temperature': 0.6}, return_dict=True))
