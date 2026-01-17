import pytest
from ArkModelNew import ArkModelLink, UserMessage


@pytest.mark.asyncio
async def test_generation_response():
    model = ArkModelLink(base_url="http://localhost:8080/v1")
    messages = [UserMessage(content="Say hello in Spanish.")]
    result = await model.make_llm_call(messages, json_schema=None)
    assert "message" in result
    assert result["message"].content.lower().startswith("hola")


@pytest.mark.asyncio
async def test_generation_with_schema():
    model = ArkModelLink(base_url="http://localhost:8080/v1")
    messages = [UserMessage(content="Give me a simple product listing.")]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["product_name", "price", "in_stock"],
        },
    }
    result = await model.make_llm_call(messages, json_schema=schema)
    schema_result = result.get("schema_result")
    assert schema_result is not None
    content = schema_result.content
    assert isinstance(content, str)
    parsed = eval(content) if isinstance(content, str) else content
    assert "product_name" in parsed
    assert "price" in parsed
    assert "in_stock" in parsed
