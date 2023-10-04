"""
Test result of this code:

('User: Hi. I like everything but fantasy films and alien type stuff. have you seen anything good lately? ', "System: Sure, if you're not into fantasy and alien films, you might enjoy The Lord of the Rings.")
"""

import sys
sys.path.append('./src')

from recwizard.modules.chatgpt import ChatgptGen, ChatgptGenConfig
from recwizard.modules.unicrs import UnicrsRec
from recwizard.modules.redial import RedialRec
from recwizard.pipelines.fill_blank import FillBlankConfig, FillBlankPipeline

if __name__ == '__main__':
    # Upload to hub if prompt is changed
    # gen_module = ChatgptGen(ChatgptGenConfig())
    # gen_module.push_to_hub('recwizard/chatgpt-gen-fillblank')
    model = FillBlankPipeline(
        config=FillBlankConfig(),
        gen_module=ChatgptGen.from_pretrained('recwizard/chatgpt-gen-fillblank'),
        # rec_module=UnicrsRec.from_pretrained('recwizard/unicrs-rec-redial'),
        rec_module=RedialRec.from_pretrained('recwizard/redial-rec'),
    ).to('cuda:0')
    query = 'User: Hi. I like everything but <entity>fantasy</entity> films and alien type stuff.' \
            ' have you seen anything good lately?'
    # print(query)
    query1 = ('User: Hi I am looking for a movie like <entity>Super Troopers (2001)</entity>'
              '<sep>System: You should watch <entity>Police Academy (1984)</entity>'
              '<sep>User: Is that a great one? I have never seen it. I have seen <entity>American Pie</entity> I mean <entity>American Pie (1999)</entity>'
              '<sep>System: Yes <entity>Police Academy (1984)</entity> is very funny and so is <entity>Police Academy 2: Their First Assignment (1985)</entity>'
              '<sep>User: It sounds like I need to check them out'
              '<sep>System: yes you will enjoy them'
              '<sep>User: I appreciate your time. I will need to check those out. Are there any others you would recommend?'
              # '<sep>System: yes <entity>Lethal Weapon (1987)<entity>'
              # '<sep>User: Thank you i will watch that too'
              # '<sep>System: and also <entity>Beverly Hills Cop (1984)</entity>'
              # '<sep>User: Thanks for the suggestions.'
              # '<sep>System: you are welcome'
              )
    print(model.response(query1, gen_args={'temperature': 0}))
