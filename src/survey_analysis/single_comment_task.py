from typing import Callable
from instructor.function_calls import OpenAISchema, Mode
from dataclasses import dataclass
from openai import AsyncOpenAI
import httpx
from .utils import comment_has_content
from abc import abstractmethod
from typing import Any, Protocol

aclient = AsyncOpenAI(timeout=httpx.Timeout(timeout=60.0))

@dataclass
class LLMConfig:
    """Model class for LLM configuration"""
    model: str = 'gpt-4-1106-preview'
    temperature: float = 0.0


# takes a string comment and returns the messages for the prompt
GetPrompt = Callable[[str], list[dict[str, str]]]

# signature for running tasks that take a string comment and return an OpenAISchema
ApplyTask = Callable[[str, GetPrompt, OpenAISchema, LLMConfig | None], OpenAISchema] 


async def apply_task(comment: str, get_prompt: GetPrompt, result_class: OpenAISchema, llm_config: LLMConfig=None) -> OpenAISchema:
    """Gets the result of applying an NLP task to a comment"""
    if llm_config is None:
        llm_config = LLMConfig()

    # if the comment has no content (is a None equivalent), return early with
    # an empty classification (filled in with defaults)
    if not comment_has_content(comment):
        return result_class()
    
    # expect partial application of get_prompt if needs something like the tags_list
    messages = get_prompt(comment)
    fn_schema = result_class.openai_schema

    response = await aclient.chat.completions.create(
        model=llm_config.model,
        temperature=llm_config.temperature,
        messages=messages,
        tools=[{"type": "function", "function": fn_schema}],
        tool_choice={"type": "function", "function": {"name": fn_schema['name']}},
    )

    # assert isinstance(expertise, Expertise)
    result = result_class.from_response(response, mode=Mode.TOOLS)

    return result


class SurveyTaskProtocol(Protocol):
    """Abstract class for a survey task"""
    @abstractmethod
    def prompt_messages(self, comment: str) -> list[dict[str, str]]:
        """Creates the messages for the prompt"""
        pass

    @abstractmethod
    def result_class(self) -> OpenAISchema:
        """Returns the result class for the task.
        The models for task results should have defaults to account for comment with no content. 
        The individual task processing routine uses the default if a comment has no content so 
        as not to incur any model costs and save latency.
        """
        pass
