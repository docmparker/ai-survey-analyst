from pydantic import BaseModel
import tiktoken

def comment_has_content(comment: str) -> bool:
    """Check if a comment has content"""
    none_equivalents = ['n/a', None, 'none', 'null', '', 'na', 'nan']
    return False if ((not comment) or (comment.lower() in none_equivalents)) else True

def count_tokens(text: str) -> int:
    """Count the number of tokens in a string"""
    enc = tiktoken.encoding_for_model('gpt-4') # the tokenizer is the same for gpt-3.5 and gpt-4
    return len(enc.encode(text))

def count_tokens_for_messages(messages, model="gpt-4"):
    """Return the number of tokens used by a list of messages."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        enc = tiktoken.get_encoding("cl100k_base")

    tokens_per_message = 3  # <|start|> <|name|> <|message|>
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for value in message.values():
            num_tokens += len(enc.encode(value))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

class OpenAISchema(BaseModel):
    """This goes from a pydantic model to an OpenAI function/tool schema"""
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