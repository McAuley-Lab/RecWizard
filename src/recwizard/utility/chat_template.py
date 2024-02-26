from typing import List, Dict, Union
from recwizard.utility import SEP_TOKEN

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ 'System: ' + message['content'] }}{% endif %}{% if not loop.last %}{{'\n'}}{% endif %}{% endfor %}"


def create_chat_message(context: Union[str, list]) -> List[Dict[str, str]]:
    def _create_chat_message(context):
        texts = context.split(SEP_TOKEN)
        messages = []
        user = "User:"
        system = "System:"
        for text in texts:
            text = text.strip()
            if text.startswith(user):
                messages.append({"role": "user", "content": text[len(user) :].strip(" ")})
            elif text.startswith(system):
                messages.append({"role": "assistant", "content": text[len(system) :].strip(" ")})
            else:
                messages.append({"role": "user", "content": text})

        return messages

    if type(context) == str:
        return _create_chat_message(context)
    elif type(context) == list:
        return [_create_chat_message(c) for c in context]


if __name__ == "__main__":
    print(create_chat_message("User: Hello; <sep> System: Hi <sep> User: How are you? <sep> System: I am fine."))
    print(create_chat_message("I am fine."))
