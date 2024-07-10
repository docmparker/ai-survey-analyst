from pydantic import BaseModel, Field, ConfigDict
from abc import ABC, abstractmethod
from .utils import OpenAISchema, comment_has_content
from typing import Type
import random

class LLMConfig(BaseModel):
    """Model class for LLM configuration"""
    model: str = 'gpt-4-0125-preview'      # 'gpt-4-1106-preview' # 'gpt-4o' also at this point
    temperature: float = 0.0
    logprobs: bool | None = None
    top_logprobs: int | None = Field(None, ge=0, le=5, description="The number of logprobs to return")


class InputModel(ABC):
    """Abstract base class for input models
    The input model has to be able to tell when it is empty so we don't need to run it through the LLM 
    in that case.
    """
    @abstractmethod
    def is_empty(self) -> bool:
        """Returns True if the input is empty"""
        pass

class CommentModel(InputModel, OpenAISchema):
    """Wraps a single comment. Used by tasks that take a single comment or a list of comments."""
    comment: str |  None = Field(None, description="The comment to process")

    model_config = ConfigDict(coerce_numbers_to_str=True) # handle nans

    def is_empty(self) -> bool:
        """Returns True if the input is empty"""
        return not comment_has_content(self.comment)

class CommentBatch(InputModel, OpenAISchema):
    """Wraps a batch of comments. Used by tasks that take a batch of comments."""
    comments: list[CommentModel] = Field([], description="A list of comments")

    def is_empty(self) -> bool:
        """Returns True if all comments are empty"""
        return all([comment.is_empty() for comment in self.comments])

    def shuffle(self) -> None:
        """Shuffles the comments"""
        random.shuffle(self.comments)

class SurveyTaskProtocol(ABC):
    """Abstract class for a survey task"""

    @abstractmethod
    def input_class(self) -> Type[InputModel]:
        """Returns the input class for the task"""
        pass

    @abstractmethod
    def prompt_messages(self, task_input: InputModel) -> list[dict[str, str]]:
        """Creates the messages for the prompt"""
        pass

    @abstractmethod
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for the task.
        The models for task results should have defaults to account for comment with no content. 
        The individual task processing routine uses the default if a comment has no content so 
        as not to incur any model costs and save latency. This could be enforced with a 
        metaclass if desired (see DefaultsEnforcedMeta), but it is not currently.
        """
        pass

class DefaultsEnforcedMeta(type(BaseModel)):
    """Enforces defaults for a result class

    Example usage:
    class MyModel(OpenAISchema, InputModel, metaclass=DefaultsEnforcedMeta):
        my_field: str = Field("default", description="A description of my field")
        another_field: int = Field(0, description="Another field with a default value")
    """
    def __new__(mcs, name, bases, attrs):
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, Field) and attr_value.default == ...:
                raise TypeError(f"Field {attr_name} must have a default value")
        return super().__new__(mcs, name, bases, attrs)
