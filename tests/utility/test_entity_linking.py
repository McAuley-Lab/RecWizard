from recwizard.utility import EntityLink


# Mocking function for load_json_file_from_dataset
def mock_load_json_file_from_dataset(dataset, fileName, dataset_org):
    # Replace this with the expected return value for testing purposes
    return {"entity1": "https://example.com/entity1", "entity2": "https://example.com/entity2"}


# Tests
def test_entity_link_init():
    entity_link = EntityLink(load_json_func=mock_load_json_file_from_dataset)
    assert entity_link.entityName2link is not None
    assert entity_link.entityName2link == {
        "entity1": "https://example.com/entity1",
        "entity2": "https://example.com/entity2",
    }


def test_entity_link_call_existing_entity():
    entity_link = EntityLink(load_json_func=mock_load_json_file_from_dataset)
    result = entity_link("entity1")
    assert result == "https://example.com/entity1"


def test_entity_link_call_nonexistent_entity():
    entity_link = EntityLink(load_json_func=mock_load_json_file_from_dataset)
    result = entity_link("nonexistent_entity")
    expected_url = "https://www.google.com/search?q=nonexistent_entity"
    assert result == expected_url
