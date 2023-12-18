import json
from pprint import pprint
import random
from instructor.function_calls import OpenAISchema
from .single_input_task import SurveyTaskProtocol, InputModel, CommentModel
from pydantic import BaseModel, Field
from typing import Type
from survey_analysis import single_input_task as sit


# Create the model
class ThematicAnalysisResult(OpenAISchema):
    """Store excerpts containing a particular goal focus extracted from a student comment"""
    excerpts: list[str] = Field([], description="A list of excerpts related to the goal focus")

class Theme(OpenAISchema):
    """Store a theme and relevant extracted quotes derived from a batch of comments"""
    title: str = Field("", description="A short name for the theme")
    description: str = Field("", description="A description of the theme")
    citations: list[str] = Field([], description="A list of citations related to the theme")

class DerivedThemes(OpenAISchema, InputModel):
    """Store the themes derived from a batch of comments"""
    themes: list[Theme] = Field([], description="A list of themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.themes) == 0

class CommentBatch(InputModel, BaseModel):
    """Wraps a batch of comments. Used by tasks that take a batch of comments."""
    comments: list[CommentModel] = Field([], description="A list of comments")

    def is_empty(self) -> bool:
        """Returns True if all comments are empty"""
        return all([comment.is_empty() for comment in self.comments])

    def shuffle(self) -> None:
        """Shuffles the comments"""
        random.shuffle(self.comments)


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



# needs an input class that is a list of DerivedThemes

class DerivedThemesBatch(InputModel, BaseModel):
    """Wraps a batch of derived themes. Used by tasks that take a batch of derived themes."""
    derived_themes: list[DerivedThemes] = Field([], description="A list of derived themes")

    def is_empty(self) -> bool:
        """Returns True if all derived themes are empty"""
        return all([derived_theme.is_empty() for derived_theme in self.derived_themes])


async def derive_themes(task_input: CommentBatch, survey_task: SurveyTaskProtocol, shuffle_passes=3) -> DerivedThemes:
    """Derives themes from a batch of comments, coordinating
    multiple shuffled passes to avoid LLM positional bias and
    then combining the results of each pass into a single result"""
    # run survey_task on task_input shuffle_passes times and combine results
    results: list[DerivedThemes] = []
    for _ in range(shuffle_passes):
        # shuffle the comments
        task_input.shuffle() # trying shuffling to see if order of comments help derive different themes
        task_result: DerivedThemes = await sit.apply_task(task_input=task_input,
                                                get_prompt=survey_task.prompt_messages,
                                                result_class=survey_task.result_class)
        pprint(json.loads(task_result.model_dump_json()))
        results.append(task_result)

    # combine the results - maybe I need to flatten the batches first and just feed this as DerivedThemes, not a batch
    reduce_task_input = DerivedThemes(themes=[theme for derived_themes in results for theme in derived_themes.themes])
    # reduce_task_input = DerivedThemesBatch(derived_themes=results)
    reduce_task = CombineThemes(survey_question=survey_task.question)
    final_task_result = await sit.apply_task(task_input=reduce_task_input,
                                            get_prompt=reduce_task.prompt_messages,
                                            result_class=reduce_task.result_class)

    return final_task_result


    
class CombineThemes(SurveyTaskProtocol):
    """Class for combining themes into fewer common themes"""
    def __init__(self, survey_question: str):
        self.survey_question = survey_question

    @property
    def input_class(self) -> Type[InputModel]:
        return DerivedThemes

    def prompt_messages(self, task_input: DerivedThemes) -> list[dict[str, str]]:
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

        def format_themes(derived_themes:DerivedThemes) -> str:
            """Formats themes into a string"""
            theme_str = "Themes:\n\n"
            # flatten the list of derived themes in the batch
            themes = [theme for theme in derived_themes.themes]
            for theme in themes:
                theme_str += f"title: {theme.title}\n"
                theme_str += f"description: {theme.description}\n"
                theme_str += f"citations: {theme.citations}\n"
                theme_str += "\n"
            return theme_str

        system_message = f"""You are a highly-skilled assistant that works with themes \
from student course feedback comments. A theme is a short phrase that summarizes \
a piece of feedback that is expressed by multiple students. You respond only with a JSON array.

You will be given a list of themes that were derived from student course feedback comments. \
Each theme has a short title, a description, and a list of citations that are exact quotes supporting \
the theme. These themes were derived from survey responses to the question: "{self.survey_question}" \
Your task is to combine similar themes into a single theme. If you find two or more \
themes that express a common idea, then combine them into a single theme with a title and \
description. When combining themes, you should combine the citations from the themes into \
a single list of citations. If any of the citations are duplicates, then remove the duplicates \
when combining the citations. Do not alter the citations in any other way. 

Once you have done your combining, respond with a JSON array of theme objects. \
Each theme object should have a title field, a description field, and a citations field."""

        user_message = f"""{format_themes(task_input)}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages


    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for theme derivation combining"""
        return DerivedThemes
