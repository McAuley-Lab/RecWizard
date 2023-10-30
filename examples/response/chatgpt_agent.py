"""
Test result of this code:

Here is three continued conversation. The code has memory and supports ChatGpt-like conversation.

('User: Hi. I like everything but fantasy films and alien type stuff. have you seen anything good lately? ', 'System: Absolutely, I recently watched a thriller movie named Gone Girl. It was quite intriguing and had no fantasy or alien-related elements. You might enjoy it! ^_^')
('User: I do not like that movie, it is too boring. Please recommend a more exciting one. ', 'System: Sure, I understand. How about the action-packed movie Die Hard? Its thrilling sequences and high octane scenes might pique your interest. ^_^')
('User: Thank you! Could you recommend one extra movie like it? ', "System: Of course! If you enjoyed the action and excitement in the previous movie, you're likely to enjoy Mad Max: Fury Road. It shares a similar thrilling pace and narrative style. ^_^")
"""

from recwizard.modules.chatgpt import ChatgptGen
from recwizard.modules.chatgpt import ChatgptRec
from recwizard.pipelines.chatgpt import ChatgptAgentConfig, ChatgptAgent

if __name__ == '__main__':
    model = ChatgptAgent(
        config=ChatgptAgentConfig(),
        prompt='All <movie> must be replaced by some phrases. No [] in your response. Finally, add a ^_^ to your response.',
        model_name='gpt-4',

        gen_module=ChatgptGen.from_pretrained('recwizard/chatgpt-gen-fillblank'),
        rec_module=ChatgptRec.from_pretrained('recwizard/chatgpt-rec-fillblank'),
    )

    query1 = 'User: Hi. I like everything but <entity>fantasy</entity> films and alien type stuff.' \
             ' have you seen anything good lately?'
    print(model.response(query1))

    query2 = 'User: I do not like that movie, it is too boring. Please recommend a more exciting one.<sep>System:'
    print(model.response(query2))

    query3 = 'Thank you! Could you recommend one extra movie like it?'
    print(model.response(query3))
