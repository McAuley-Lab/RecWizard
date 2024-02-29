import re

from typing import List, Dict, Union
from recwizard.utils import SEP_TOKEN, START_TAG, END_TAG, ASSISTANT_TOKEN, USER_TOKEN


def create_chat_message(context: Union[str, list]) -> List[Dict[str, str]]:
    def _create_chat_message(context):
        texts = context.split(SEP_TOKEN)
        messages = []
        for text in texts:
            text = text.strip()
            if text.startswith(USER_TOKEN):
                messages.append({"role": "user", "content": text[len(USER_TOKEN) :].strip(" ")})
            elif text.startswith(ASSISTANT_TOKEN):
                messages.append({"role": "assistant", "content": text[len(ASSISTANT_TOKEN) :].strip(" ")})
            else:
                messages.append({"role": "user", "content": text})

        return messages

    if type(context) == str:
        return _create_chat_message(context)
    elif type(context) == list:
        return [_create_chat_message(c) for c in context]


def create_item_list(text: str) -> List[str]:
    """Return a list of entities from the given text

    Args:
        text (str): The text to extract entities from, e.g., "<entity>Jumanji_(2017_sequel)</entity> <entity>Bridesmaids_(2011_film)</entity> <entity>Bad_Moms</entity>"

    Returns:
        List[str]: A list of entities, e.g., ["Jumanji_(2017_sequel)", "Bridesmaids_(2011_film)", "Bad_Moms"]
    """
    entities = re.findall(rf"{START_TAG}(.*?){END_TAG}", text)
    return entities


if __name__ == "__main__":
    print(create_chat_message("User: Hello; <sep> System: Hi <sep> User: How are you? <sep> System: I am fine."))
    print(create_chat_message("I am fine."))

    print(
        create_item_list(
            "<entity>Jumanji_(2017_sequel)</entity> <entity>Bridesmaids_(2011_film)</entity> <entity>Bad_Moms</entity>"
        )
    )
