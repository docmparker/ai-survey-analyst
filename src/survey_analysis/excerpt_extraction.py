from .utils import OpenAISchema
from .models_common import InputModel, SurveyTaskProtocol, CommentModel
from .single_input_task import apply_task
from pydantic import Field, validate_arguments
from typing import Type
from functools import partial
from . import batch_runner as br


# Create the model - here we do it outside the class so it can also be used elsewhere if desired
# without instantiating the class
class ExcerptExtractionResult(OpenAISchema):
    """Store excerpts containing a particular goal focus extracted from a student comment"""
    excerpts: list[str] = Field([], description="A list of excerpts related to the goal focus")


class ExcerptExtraction(SurveyTaskProtocol):
    """Class for excerpt extraction"""
    def __init__(self, goal_focus: str, question: str):
        self.goal_focus = goal_focus
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel

    def prompt_messages(self, task_input: CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the extraction prompt"""

        delimiter = "####"
        # question = "What could be improved about the course?"
        # goal_focus = "suggestions for improvement"

        system_message = f"""You are an assistant that extracts {self.goal_focus} from \
student course feedback comments.  You respond only with a JSON array.

You will be provided with a comment from a student course feedback survey. \
The comment will be delimited by {delimiter} characters. \
Each original comment was in response to the question: "{self.question}". \
However, your task is to only select excerpts which pertain to the goal focus: "{self.goal_focus}". \
Excerpts should only be exact quotes taken from the comment; do not add or alter words \
under any circumstances.

If you cannot extract excerpts for any reason, for example if the comment is \
not relevant to the question, respond only with an empty JSON array: []

If there are relevant excerpts, ensure that excerpts contain all relevant text needed to interpret them - \
in other words don't extract small snippets that are missing important context.

Before finalizing excerpts, review your excerpts to see if any consecutive excerpts \
are actually about the same suggestion or part of the same thought. If so, combine them \
into a single excerpt."""

        user_message = f"""{delimiter}{task_input.comment}{delimiter}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for extraction"""
        return ExcerptExtractionResult
    

# @validate_arguments
# async def extract_excerpts(*, comment: str, question: str, goal_focus: str) -> OpenAISchema:
#     """Extract excerpts containing a particular goal focus from a student comment"""
#     survey_task = ExcerptExtraction(goal_focus, question)
#     task_input = CommentModel(comment=comment)
#     result = await apply_task(task_input=task_input,
#                                         get_prompt=survey_task.prompt_messages,
#                                         result_class=survey_task.result_class)
#     return result


# TODO: consider making this a class method
@validate_arguments
async def extract_excerpts(*, comments: list[str | None], question: str, goal_focus: str) -> list[OpenAISchema]:
    """Extract excerpts from a list of comments, based on a particular question and goal_focus
    
    Returns a list of ExcerptExtractionResult objects
    """

    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    survey_task: SurveyTaskProtocol = ExcerptExtraction(goal_focus=goal_focus, question=question)
    ex_task = partial(apply_task, get_prompt=survey_task.prompt_messages, result_class=survey_task.result_class)
    extractions = await br.process_tasks(comments_to_test, ex_task)

    return extractions