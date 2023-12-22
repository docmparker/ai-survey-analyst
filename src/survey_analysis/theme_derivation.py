import json
from pprint import pprint
import random
from instructor.function_calls import OpenAISchema
from .single_input_task import SurveyTaskProtocol, InputModel, CommentModel
from pydantic import BaseModel, Field
from typing import Type
from survey_analysis import single_input_task as sit


# Create the models

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

class ThemeConsolidation(OpenAISchema, InputModel):
    """Store the results from the process of combining a list of themes derived from a batch of comments"""
    step1_reasoning: str = Field("", description="The reasoning for combining themes")
    step2_intermediate_themes: DerivedThemes = Field(..., description="The intermediate themes after combining similar themes")
    final_combined_themes: list[Theme] = Field(..., description="The final list of all consolidated themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.final_combined_themes) == 0


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
Each theme object should have a 'title' field (which \
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


async def derive_themes(task_input: CommentBatch, survey_task: SurveyTaskProtocol, shuffle_passes=3) -> ThemeConsolidation:
    """Derives themes from a batch of comments, coordinating
    multiple shuffled passes to avoid LLM positional bias and
    then combining the results of each pass into a single result"""
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
            print(f"title: {theme.title}")
            print(f"description: {theme.description}")
            print(f"citations: {theme.citations}")
            print()
        # pprint(json.loads(task_result.model_dump_json()))

        results.extend(task_result.themes)

    print(f"number of total themes across {shuffle_passes} passes: {len(results)}")

    reduce_task_input = DerivedThemes(themes=results)
    reduce_task = CombineThemes(survey_question=survey_task.question)
    print("combining results")
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
                theme_str += f"title: {theme.title}\n"
                theme_str += f"description: {theme.description}\n"
                theme_str += f"citations:\n"
                for citation in theme.citations:
                    theme_str += f"  - {citation}\n"
                theme_str += f"</theme>\n"
                theme_str += "\n"
            return theme_str

        system_message = f"""You are an assistant who is highly skilled at working with \
student feedback surveys. Your task is to thoroughly analyze and consolidate a list \
of themes derived from student feedback on an online course. Each theme consists of a \
'title', 'description', and 'citations' (a list of exact quotes from students). These \
themes were derived from survey responses to the question: "{self.survey_question}". \
Your primary focus is on merging themes that cover the same or very similar topics, \
based on their titles and descriptions.

First, identify similar themes. Review each theme and compare with others to find content \
similarities. Look for themes with overlapping or complementary content, even if their titles \
are not identical. For instance, 'Expert Instruction' and 'Expert Instructors' might \
cover similar content from different angles and should be considered for merging. Similarly, \
'Interactive Learning' and 'Hands-On Modules' may also overlap significantly in content. \
Save your detailed reasoning from this first step for later output in a JSON object, under the \
key 'step1_reasoning'.

Next, merge and refine themes: Having identified similar themes:
   - Combine their citations into one list, removing any duplicates.
   - Create a new, consolidated title that captures the essence of the merged themes.
   - Write a comprehensive description that encompasses all aspects of the themes being merged.

Save the output of this step for later output in a JSON object, under the key \
'step2_intermediate_themes', as a list of themes including the 'title', \
'description', and 'citations' for each theme.

Finally, look for any unique themes from the original list of themes that may have been lost \
in the consolidation and, if necessary, update the consolidated list of themes to include \
those. If no themes were lost, just repeat the whole list of themes for the output of this step. \
Save the final, consolidated list of themes for later output in a JSON object under the \
key 'final_combined_themes' as a list of theme objects, each containing title, description, and citations.

After you have completed all of these steps, present the final overall results in a \
JSON object, including 'step1_reasoning', 'step2_intermediate_themes', and 'final_combined_themes'.

You will now be presented with the original list of themes."""

        user_message = f"""{format_themes(task_input)}"""

        messages =  [  
            {'role':'system', 'content': system_message},
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for theme derivation combining"""
        return ThemeConsolidation
