"""
Test result of this code:

('User: Hi. I like everything but fantasy films and alien type stuff. have you seen anything good lately? ', "System: Sure, if you're not into fantasy and alien films, you might enjoy The Lord of the Rings.")
"""

from recwizard.modules.chatgpt import ChatgptGen, ChatgptGenConfig
from recwizard.modules.unicrs import UnicrsRec
from recwizard.modules.redial import RedialRec
from recwizard import ExpansionConfig, ExpansionPipeline

prompt = {
    'role': 'system',
    'content': (  
        ' You are a customer support recommending movies to the user.'
        ' Our operating platform returns suggested movies in real time from the dialogue history.'
        ' You may choose from the suggestions and elaborate on why the user may like them.'
        ' Or you can choose to reply without a recommendation.'
        ' Now The system is suggesting the following movies: {}.'
        # ' Carefully review the dialogue history before I write a response to the user.'
        ' If a movie comes in the format with a year, e.g. ""Inception (2010)"",'
        ' you should see the year (2010) as a part of the movie name.'
        ' You should not use the format ""Inception" (2010)" by leaving the year out of the quotation mark.'
        ' You should keep in mind that the system suggestion is only for reference.'
        # ' If the user is saying things like thank you or goodbye,'
        ' You should prioritize giving a quick short response over throwing more movies at the user,'
        ' especially when the user is likely to be leaving.'
        ' You should not not respond to this message.'
    )
}

if __name__ == '__main__':


    # upload to hub if prompt is changed
    # module = ChatgptGen(ChatgptGenConfig(prompt=prompt))
    # module.push_to_hub('recwizard/chatgpt-gen-expansion')

    model = ExpansionPipeline(
        config=ExpansionConfig(),
        gen_module=ChatgptGen.from_pretrained('recwizard/chatgpt-gen-expansion'),
        # rec_module=UnicrsRec.from_pretrained('recwizard/unicrs-rec-redial'),
        rec_module=RedialRec.from_pretrained('recwizard/redial-rec'),
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
    print(model.response(query2, rec_args={'topk': 10}, gen_args={'temperature': 0}, return_dict=True))
