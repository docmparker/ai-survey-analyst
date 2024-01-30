import json
from pprint import pprint
import random
# from instructor.function_calls import OpenAISchema
from .utils import OpenAISchema
from .single_input_task import SurveyTaskProtocol, InputModel, CommentModel
from pydantic import BaseModel, Field, validate_arguments, conint
from typing import Type
from survey_analysis import single_input_task as sit


# Create the models

class Theme(OpenAISchema):
    """A theme and relevant extracted quotes derived from a batch of comments"""
    theme_title: str = Field("", description="A short name for the theme")
    description: str = Field("", description="A description of the theme")
    citations: list[str] = Field([], description="A list of citations (exact extracted quotes) related to the theme")

class DerivedThemes(OpenAISchema, InputModel):
    """Store the themes derived from a batch of comments"""
    themes: list[Theme] = Field([], description="A list of themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.themes) == 0

class CommentBatch(InputModel, OpenAISchema):
    """Wraps a batch of comments. Used by tasks that take a batch of comments."""
    comments: list[CommentModel] = Field([], description="A list of comments")

    def is_empty(self) -> bool:
        """Returns True if all comments are empty"""
        return all([comment.is_empty() for comment in self.comments])

    def shuffle(self) -> None:
        """Shuffles the comments"""
        random.shuffle(self.comments)

# could make combine_themes have its field be a list of reasoning and combined_themes and let it decide
# how many to include. If these are classes refer in the system prompt as Reasoning and CombinedThemes
class Reasoning(OpenAISchema, InputModel):
    """The reasoning for combining themes"""
    reasoning: str = Field(..., description="The reasoning for combining themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.reasoning) == 0
    
# this is basically the same class as DerivedThemes, but with a different name and description
# to potentially enhance the tool use based on a more descriptive schema for this task
class UpdatedThemes(OpenAISchema, InputModel):
    """Updated themes after combining similar themes, including merged themes and themes that didn't need merging"""
    themes: list[Theme] = Field([], description="A list of themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.themes) == 0

# This gets converted to the function/tool schema for combining themes, hence the non-classy name
class combine_themes(OpenAISchema, InputModel):
    """Consolidates a list of themes by combining very similar themes and keeping unique themes"""
    reasoning: str = Field(..., description="The reasoning for combining themes")
    updated_themes: UpdatedThemes = Field(..., description="The updated themes after combining similar themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.updated_themes) == 0


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
Each theme object should have a 'theme_title' field (which \
gives a short name for the theme in 5 words or less), a 'description' field which describes the \
theme in two sentences or less, and a 'citations' \
field, which is an array of 3 exact quotes from distinct survey comments \
supporting this theme. Each quote should have enough context to be understood. \
Do not add or alter words in the quotes under any circumstances. If there are \
less than 3 quotes, then include as many as you can."""

        comment_list_del = "\n".join([f"{delimiter}{comment.comment}{delimiter}" 
                                      for comment in task_input.comments 
                                      if not comment.is_empty()])
        
        user_message = f"""{comment_list_del}"""

        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]

        return messages
    
    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for theme derivation"""
        return DerivedThemes

# turn this into a pipeline
# async def derive_themes(task_input: CommentBatch, survey_task: DeriveThemes, shuffle_passes=3) -> ThemeConsolidation:
@validate_arguments
async def derive_themes(comments: list[str | None], question: str, shuffle_passes: conint(ge=1, le=10) = 3) -> combine_themes:
    """Derives themes from a batch of comments, coordinating
    multiple shuffled passes to avoid LLM positional bias and
    then combining the results of each pass into a single result.

    Args: 
        comments: A list of comments
        question: The survey question that the comments are in response to
        shuffle_passes: The number of times to shuffle the comments and derive themes (default 3
    
    Example usage:
    ```
    question = "What were the best parts of the course?"
    comments = sanitized_survey['best_parts'].tolist()[:100]
    sample_output = await derive_themes(comments=comments, question=question)
    ```
    """

    survey_task: DeriveThemes = DeriveThemes(question=question)
    task_input: CommentBatch = CommentBatch(comments=[CommentModel(comment=comment) for comment in comments])
    
    # run survey_task on task_input shuffle_passes times and combine results
    results: list[DerivedThemes] = []
    for i in range(shuffle_passes):
        # shuffle the comments
        print(f"shuffle pass {i}")
        task_input.shuffle() # trying shuffling to see if order of comments help derive different themes
        task_result: DerivedThemes = await sit.apply_task(task_input=task_input, 
                                                          get_prompt=survey_task.prompt_messages, 
                                                          result_class=survey_task.result_class)
        # show the theme titles and descriptions
        for theme in task_result.themes:
            print(f"title: {theme.theme_title}")
            print(f"description: {theme.description}")
            print(f"citations: {theme.citations}")
            print()

        results.extend(task_result.themes)

    print(f"number of total themes across {shuffle_passes} passes: {len(results)}")

    # if there was only one pass, return the result of that pass
    if shuffle_passes == 1:
        return results[0]
    
    reduce_task_input = DerivedThemes(themes=results)
    reduce_task = CombineThemes(survey_question=survey_task.question)
    print("\ncombining results\n")
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
        """Creates the messages for the theming prompt"""

        def format_themes(derived_themes:DerivedThemes) -> str:
            """Formats themes into a string"""
            # theme_str = "Themes:\n\n"
            theme_str = ""
            # flatten the list of derived themes in the batch
            for theme in derived_themes.themes:
                theme_str += f"<theme>\n"
                theme_str += f"title: {theme.theme_title}\n"
                theme_str += f"description: {theme.description}\n"
                theme_str += f"citations:\n"
                for citation in theme.citations:
                    theme_str += f"  - {citation}\n"
                theme_str += f"</theme>\n"
                theme_str += "\n"
            return theme_str

        system_message = f"""You are an assistant who is highly skilled at working with \
student feedback surveys. You will receive a list of themes. Your task is to merge \
themes that cover the same or very similar topics, based on theme titles and descriptions. \
Each theme consists of a 'theme_title', 'description', and 'citations' (a list of exact quotes from \
students). These themes were derived from survey responses to the question: "{self.survey_question}". 

Step 1:

First, identify similar themes. Review each theme and compare with others to find content \
similarities. Look for themes with overlapping or complementary content, even if their titles \
are not identical. For instance, 'Expert Instruction' and 'Expert Instructors' might \
cover similar content from different angles and should be considered for merging. Similarly, \
'Interactive Learning' and 'Hands-On Modules' may also overlap significantly in content. \
Record your reasoning from this step.

Step 2:

Next, merge and refine themes. Having identified similar themes in the previous step:
   - Combine their citations into one list, removing any exact duplicate citations.
   - For each merged theme, create a new, consolidated title that captures the essence of the merged theme.
   - For each merged theme, write a comprehensive description that encompasses all aspects of the themes being merged.

For each unique theme that doesn't need merging, leave it as is.

Save the updated themes from this step. Include all of the new merged ones and the ones that didn't need \
merging. For every theme, include the 'theme_title', 'description', and 'citations'.

Step 3:

Review your work. Did you do an excellent job? If you missed combining any similar themes, \
combine those before saving the final themes. As before, don't forget to remove any \
duplicate citations in each list of citations. Do your best. If you have done an outstanding \
job, you will get a $500 tip."""

        user_message = f"""{format_themes(task_input)}"""

        messages =  [  
            {'role':'system', 'content': system_message},
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for theme derivation combining"""
        return combine_themes
