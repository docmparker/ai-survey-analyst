from typing import Any, Type
from pydantic import BaseModel


def comment_has_content(comment: str) -> bool:
    """Check if a comment has content"""
    none_equivalents = ['n/a', None, 'none', 'null', '', 'na']
    return False if ((not comment) or (comment.lower() in none_equivalents)) else True


class OpenAISchema(BaseModel):
    @classmethod
    def openai_schema(cls):
        assert cls.__doc__, f"{cls.__name__} is missing a docstring."
        assert (
            "title" not in cls.model_fields
        ), "`title` is a reserved keyword and cannot be used as a field name."
        schema_dict = cls.model_json_schema()
        cls.remove_a_key(schema_dict, "title")

        return {
            "name": cls.__name__,
            "description": cls.__doc__,
            "parameters": schema_dict,
        }

    @classmethod
    def remove_a_key(cls, d, remove_key):
        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == remove_key:
                    del d[key]
                else:
                    cls.remove_a_key(d[key], remove_key)