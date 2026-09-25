import os

import pytest


def test_schema_fallback_execution():
    """Test executing schema.py with pydantic missing to cover fallback class definitions."""
    schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine", "schema.py"))
    with open(schema_path, "r", encoding="utf-8") as f:
        code = f.read()

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "pydantic":
            raise ImportError("No module named pydantic")
        return real_import(name, *args, **kwargs)

    scope = {"__builtins__": {**__builtins__, "__import__": mock_import}}
    compiled = compile(code, schema_path, "exec")
    exec(compiled, scope)

    assert scope["HAS_PYDANTIC"] is False
    BaseModel = scope["BaseModel"]
    Field = scope["Field"]
    field_validator = scope["field_validator"]
    FillerWordMetric = scope["FillerWordMetric"]

    # Test Field
    assert Field(default="val") == "val"
    assert Field() is None

    # Test field_validator
    @field_validator("some_field")
    def dummy(cls, v):
        return v

    assert dummy(None, 123) == 123

    # Test BaseModel
    bm = BaseModel(name="test", score=10)
    assert bm.name == "test"
    assert bm.score == 10

    # Test model_dump with list and nested BaseModel
    child = BaseModel(c_id=99)
    parent = BaseModel(child=child, arr=[child, 42], name="p")
    dumped = parent.model_dump()
    assert dumped["name"] == "p"
    assert dumped["child"] == {"c_id": 99}
    assert dumped["arr"] == [{"c_id": 99}, 42]

    # Test model_validate
    val_bm = BaseModel.model_validate({"a": 1, "b": "xyz"})
    assert val_bm.a == 1
    assert val_bm.b == "xyz"

    # Test FillerWordMetric fallback
    fwm = FillerWordMetric(token="umm", count=4)
    assert fwm.token == "umm"
    assert fwm.count == 4
