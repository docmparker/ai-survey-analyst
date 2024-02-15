from typing import Callable
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx
from .utils import OpenAISchema
from .models_common import LLMConfig, InputModel

aclient = AsyncOpenAI(timeout=httpx.Timeout(timeout=240.0))

# takes a task object and returns the messages for the prompt
GetPrompt = Callable[[BaseModel], list[dict[str, str]]]

# signature for running tasks that take a task object and return an OpenAISchema
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
    if task_input.is_empty():
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
    if task_input.is_empty():
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