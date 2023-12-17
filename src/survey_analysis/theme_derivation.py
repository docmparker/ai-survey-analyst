from instructor.function_calls import OpenAISchema
from .single_input_task import SurveyTaskProtocol, InputModel, CommentModel
from pydantic import BaseModel, Field
from typing import Type


# Create the model
class ThematicAnalysisResult(OpenAISchema):
    """Store excerpts containing a particular goal focus extracted from a student comment"""
    excerpts: list[str] = Field([], description="A list of excerpts related to the goal focus")

    # store_comment_themes_function = {
    #     "name": "store_comment_themes",
    #     "description": f"Store the the main themes of a batch of comments in a database.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "themes": {
    #                 "type": "array",
    #                 "items": {
    #                     "type": "object",
    #                     "description": f"A theme that applies to one or more comments.",
    #                     "properties": {
    #                         "theme_description": {
    #                             "type": "string",
    #                             "description": f"A description of the theme.",
    #                         },
    #                         "theme_title": {
    #                             "type": "string",
    #                             "description": f"A short name for the theme.",
    #                         },
    #                     },
    #                 },
    #                 "description": "A list of themes.",
    #             },
    #         },
    #         "required": ["themes"],
    #     },
    # }

class Theme(OpenAISchema):
    """Store a theme and relevant extracted quotes derived from a batch of comments"""
    title: str = Field("", description="A short name for the theme")
    description: str = Field("", description="A description of the theme")
    citations: list[str] = Field([], description="A list of citations related to the theme")

class DerivedThemes(OpenAISchema):
    """Store the themes derived from a batch of comments"""
    themes: list[Theme] = Field([], description="A list of themes")

class CommentBatch(InputModel, BaseModel):
    """Wraps a batch of comments. Used by tasks that take a batch of comments."""
    comments: list[CommentModel] = Field([], description="A list of comments")

    def is_empty(self) -> bool:
        """Returns True if all comments are empty"""
        return all([comment.is_empty() for comment in self.comments])


# TODO: give this a different protocol to distinguish it from tasks that take a single comment
# TODO: make themes in a more robust format than a string since this could be a result class from some 
# other task
class DeriveThemes(SurveyTaskProtocol):
    """Class for deriving themes from a batch of comments"""
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentBatch

    def prompt_messages(self, task_input: CommentBatch) -> list[dict[str, str]]:
        """Creates the messages for the theming prompt"""

        delimiter = "####"

        # If you find two or more \
        # comments that express a common idea, then combine them into a single theme. 

        system_message = f"""You are an assistant that derives themes from student course \
feedback comments.  You respond only with a JSON array.

You will be provided with a batch of comments from a student course feedback survey. \
Each comment is surrounded by the delimiter {delimiter} \
Each original comment was in response to the question: "{self.question}". \
Your task is to derive themes from the comments.  A theme is a short phrase that summarizes \
a piece of feedback that is expressed by multiple students. There may be multiple themes \
present in the comments. Examples of themes are: "Helpful Videos", "Clinical Applications", \
and "Interactive Content".

Once you have derived the themes, respond with a JSON array of theme objects. \
Each theme object should have a 'theme_description' field which describes the \
theme in two sentences or less, a 'theme_title' field (which \
gives a short name for the theme in 5 words or less), and a 'citations' \
field, which is an array of 3 exact quotes from distinct survey comments \
supporting this theme. Each quote should have enough context to be understood. \
Do not add or alter words in the quotes under any circumstances. If there are \
less than 3 quotes, then include as many as you can."""

        comment_list_del = "\n".join([f"{delimiter}{comment.comment}{delimiter}" for comment in task_input.comments if not comment.is_empty()])
        
        user_message = f"""{comment_list_del}"""

        messages =  [
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages
    
    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for theme derivation"""
        return DerivedThemes


class CombineThemes(SurveyTaskProtocol):
    """Class for combining themes into fewer common themes"""
    def __init__(self, goal_focus: str, question: str):
        self.goal_focus = goal_focus
        self.question = question
        self._result_class = None

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel

    def prompt_messages(self, themes: str, survey_question: str) -> list[dict[str, str]]:
        """Creates the messages for the theming prompt
        Expects themes to be of form:
        ('Themes:\n'
        '\n'
        'Theme title: Helpful Videos\n'
        'Theme description: Many students appreciated the use of videos in the '
        'course, finding them helpful for understanding complex concepts.\n'
        '\n'
        'Theme title: Clinical Applications\n'
        'Theme description: Students found the clinical applications and real-world '
        'examples particularly engaging and useful for understanding the material.\n'
        '\n')
        """

        system_message = f"""You are a highly-skilled assistant that works with themes \
from student course feedback comments. A theme is a short phrase that summarizes \
a piece of feedback that is expressed by multiple students. You respond only with a JSON array.

You will be given a list of themes that were derived from student course feedback comments. \
These themes were derived from responses to the question: "{survey_question}". \
Your task is to combine similar themes into a single theme.  If you find two or more \
themes that express a common idea, then combine them into a single theme.

Once you have done your combining, respond with a JSON array of theme objects. \
Each theme object should have a 'theme_description' field which describes the \
theme in two sentences or less, a 'theme_title' field (which \
gives a short name for the theme in 5 words or less), and a 'combined_themes' \
field (which is an array of the theme titles that were combined into this theme)."""

        user_message = f"""{themes}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages


    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for thematic analysis, dynamically creating it if necessary"""

        return OpenAISchema
