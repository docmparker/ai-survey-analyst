import random
from typing import Callable
from pydantic import BaseModel, Field, ConfigDict
from openai import AsyncOpenAI
import httpx
from .utils import comment_has_content, OpenAISchema
from abc import ABC, abstractmethod
from typing import Type

aclient = AsyncOpenAI(timeout=httpx.Timeout(timeout=240.0))

class LLMConfig(BaseModel):
    """Model class for LLM configuration"""
    model: str = 'gpt-4-0125-preview'      # 'gpt-4-1106-preview'
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
    comment: str | None = Field(None, description="The comment to process")

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


# takes a task object and returns the messages for the prompt
GetPrompt = Callable[[BaseModel], list[dict[str, str]]]

# signature for running tasks that take a task object and returns an OpenAISchema
ApplyTask = Callable[[BaseModel, GetPrompt, OpenAISchema, LLMConfig | None], OpenAISchema] 


async def apply_task(task_input: InputModel, 
                     get_prompt: GetPrompt, 
                     result_class: OpenAISchema, 
                     llm_config: LLMConfig=None) -> OpenAISchema:
    """Gets the result of applying an NLP task to a comment, list of comments, or some other unit or work."""
    if llm_config is None:
        llm_config = LLMConfig()

    # if the task_input has no content (is a None equivalent), return early with
    # an empty classification (filled in with defaults)
    if task_input.is_empty():   # not comment_has_content(comment):
        return result_class()
    
    # expect partial application of get_prompt if needs something like the tags_list
    messages = get_prompt(task_input)
    fn_schema = result_class.openai_schema()

    response = await aclient.chat.completions.create(
        model=llm_config.model,
        temperature=llm_config.temperature,
        messages=messages,
        tools=[{"type": "function", "function": fn_schema}],
        tool_choice={"type": "function", "function": {"name": fn_schema['name']}},
    )

    # assert isinstance(expertise, Expertise)
    # result = result_class.from_response(response, mode=Mode.TOOLS)
    args = response.choices[0].message.tool_calls[0].function.arguments
    result = result_class.model_validate_json(args)

    return result

async def apply_task_with_logprobs(task_input: InputModel, 
                     get_prompt: GetPrompt, 
                     result_class: OpenAISchema, 
                     llm_config: LLMConfig=None) -> OpenAISchema:
    """Gets the result of applying an NLP task to a comment, list of comments, or some other unit or work."""
    if llm_config is None:
        llm_config = LLMConfig()

    # if the task_input has no content (is a None equivalent), return early with
    # an empty classification (filled in with defaults)
    if task_input.is_empty():   # not comment_has_content(comment):
        return result_class()
    
    # expect partial application of get_prompt if needs something like the tags_list
    messages = get_prompt(task_input)
    # fn_schema = result_class.openai_schema()

    response = await aclient.chat.completions.create(
        model=llm_config.model,
        temperature=llm_config.temperature,
        messages=messages,
        # tools=[{"type": "function", "function": fn_schema}],
        # tool_choice={"type": "function", "function": {"name": fn_schema['name']}},
        response_format={"type": "json_object"},
        logprobs=llm_config.logprobs,
        top_logprobs=llm_config.top_logprobs
    )

    args = response.choices[0].message.content 

    # take the json output and put it in the result class
    result = result_class.model_validate_json(args) 

    # add the logprobs to the result class
    logprobs = response.choices[0].logprobs
    result.logprobs = logprobs

    return result

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

