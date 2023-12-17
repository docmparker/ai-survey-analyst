from instructor.function_calls import OpenAISchema
from .utils import comment_has_content
from .single_input_task import InputModel, SurveyTaskProtocol, CommentModel
from pydantic import BaseModel, Field
from typing import Type


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
        """Returns the result class for multilabel classification"""
        return ExcerptExtractionResult
    
