import urllib.parse


class EntityLink:
    entityName2link = None

    def __call__(self, entityName):
        return f"https://www.google.com/search?q={urllib.parse.quote_plus(entityName)}"
