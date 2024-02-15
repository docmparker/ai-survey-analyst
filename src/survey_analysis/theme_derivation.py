from collections import Counter
from .utils import OpenAISchema
from .models_common import SurveyTaskProtocol, InputModel, CommentModel, CommentBatch
from pydantic import Field, validate_arguments, conint
from typing import Type
from survey_analysis import single_input_task as sit
from functools import partial


# Create the models

class Theme(OpenAISchema):
    """A theme and relevant extracted quotes derived from a batch of comments"""
    theme_title: str = Field("", description="A short name for the theme (5 words or less)")
    description: str = Field("", description="A description of the theme (2 sentences or less)")
    citations: list[str] = Field([], description="A list of citations (exact extracted quotes) related to the theme")

class DerivedThemes(OpenAISchema, InputModel):
    """Store the themes derived from a batch of comments"""
    themes: list[Theme] = Field([], description="A list of themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.themes) == 0

    
# this is basically the same class as DerivedThemes, but with a different name and description
# to potentially enhance the tool use based on a more descriptive schema for this task
# class UpdatedThemes(OpenAISchema, InputModel):
#     """Updated themes after combining similar themes, including merged themes and themes that didn't need merging"""
#     themes: list[Theme] = Field([], description="A list of themes")

#     def is_empty(self) -> bool:
#         """Returns True if all themes are empty"""
#         return len(self.themes) == 0

# This gets converted to the function/tool schema for combining themes, hence the non-classy name
class combine_themes(OpenAISchema, InputModel):
    """Consolidates a list of themes by combining very similar or identical themes and keeping unique themes"""
    reasoning: str = Field(..., description="The reasoning for combining themes")
    updated_themes: list[Theme] = Field(..., description="The updated themes after combining similar themes")

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
Your goal is to derive as many themes as possible from the comments. \
A theme is a short phrase that summarizes a piece of feedback that is expressed by multiple \
students. Examples of themes are: "Helpful Videos", "Clinical Applications", and "Interactive Content". \
The themes you derive should be unique (in other words, be distinct from each other in terms \
of the feedback they represent) and comprehensive (in other words, encompass ALL feedback that \
is expressed by two or more students).

Once you have derived the themes, respond with a JSON array of theme objects. \
Each theme object should have a 'theme_title' field (which gives a short name for \
the theme in 5 words or less), a 'description' field which describes the \
theme in two sentences or less, and a 'citations' field, which is an array of \
3 exact quotes from distinct survey comments supporting this theme. Each quote \
should have enough context to be understood. Do not add or alter words in the \
quotes under any circumstances. If there are less than 3 quotes, then include \
as many as you can."""

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


@validate_arguments
async def combine_themes_(themes: list[Theme], question: str) -> combine_themes:
    """Convenience method for combining themes derived from a batch of comments
    
    question refers to the question that the comments were in response to.
    themes are the themes to combine, potentially derived from multiple passes of derive_themes,
    or from single passes over multiple batches of comments.
    """

    task_input = DerivedThemes(themes=themes)
    reduce_task = CombineThemes(survey_question=question)
    combined_result = await sit.apply_task(task_input=task_input,
                                            get_prompt=reduce_task.prompt_messages,
                                            result_class=reduce_task.result_class)

    return combined_result


@validate_arguments
async def derive_themes(comments: list[str | float | None], question: str, shuffle_passes: conint(ge=1, le=10) = 3) -> combine_themes:
    """Derives themes from a batch of comments, coordinating
    multiple shuffled passes to avoid LLM positional bias and
    then combining the results of each pass into a single result.
    Each pass is progressively combined with the combined results of the previous passes, 
    in an iterative fashion to keep tokens and task complexity relatively low.

    Args: 
        comments: A list of comments
        question: The survey question that the comments are in response to
        shuffle_passes: The number of times to shuffle the comments and derive themes (default 3, 
                            minimum 1, maximum 10)
    
    Example usage:
    ```
    question = "What were the best parts of the course?"
    comments = sanitized_survey['best_parts'].tolist()[:100]
    sample_output = await derive_themes(comments=comments, question=question)
    ```
    """

    # run derive_themes_task on task_input shuffle_passes times, combining the results each time
    # combine results of pass 1 and 2, then combine pass 3 with that, etc.

    # some helper functions
    def deduplicate_citations(themes: list[Theme]) -> list[Theme]:
        """Deduplicates the citations in a list of themes"""
        duplicate_citations = 0
        for theme in themes:
            unique_citations = set(theme.citations)
            duplicate_citations += len(theme.citations) - len(unique_citations)
            theme.citations = list(unique_citations)
        print(f"The total number of duplicate citations was {duplicate_citations}")
        return themes

    def deduplicate_themes_by_title(themes: list[Theme]) -> list[Theme]:
        """Deduplicates themes by title"""
        deduped_results = []
        for theme in themes:
            if theme.theme_title not in [t.theme_title for t in deduped_results]:
                deduped_results.append(theme)
        return deduped_results

    def show_themes(themes: list[Theme]) -> None:
        """Prints the titles and descriptions of a list of themes"""
        for theme in themes:
            print(f"title: {theme.theme_title}")
            print(f"description: {theme.description}")
            print()

    running_results = []

    # first run
    print("pass 1")
    derive_themes_task: DeriveThemes = DeriveThemes(question=question)
    comments_wrapped = [CommentModel(comment=comment) for comment in comments]
    task_input: CommentBatch = CommentBatch(comments=comments_wrapped)
    derive_partial = partial(sit.apply_task, 
                             get_prompt=derive_themes_task.prompt_messages, 
                             result_class=derive_themes_task.result_class)
    task_result: DerivedThemes = await derive_partial(task_input=task_input) 
    running_results.extend(task_result.themes)

    show_themes(running_results)

    # if there was only one pass, return the result of that pass wrapped in combine_themes
    if shuffle_passes == 1:
        return combine_themes(reasoning="only one pass", updated_themes=running_results)

    # do the remaining runs, combining progressively along the way
    for i in range(1, shuffle_passes):
        # shuffle the comments
        print(f"pass {i+1}")
        task_input.shuffle() # the themes derived vary based on the order of the comments
        task_result: DerivedThemes = await derive_partial(task_input=task_input)
                                                        #   get_prompt=derive_themes_task.prompt_messages, 
                                                        #   result_class=derive_themes_task.result_class)

        show_themes(task_result.themes)

        # add to the running results and then run a combination step
        running_results.extend(task_result.themes)

        # put out some diagnostics
        print(f"number of total themes across {i+1} passes before combining: {len(running_results)}")
        print(f"The number of unique themes by title is {len(set([theme.theme_title for theme in running_results]))}")
        theme_counts = Counter([theme.theme_title for theme in running_results])
        print(f"The count of themes by title is {theme_counts}")

        # now combine the results to distill them down to unique themes
        print(f"\ncombining results {i+1} with {i}\n")
        combined_result = await combine_themes_(themes=running_results, question=question)

        running_results = combined_result.updated_themes
        # put out some diagnostics
        print(f"number of themes after combining: {len(running_results)}")
        print(f"theme titles after combining: {[theme.theme_title for theme in running_results]}") 

    # only dedupe by titles at the very end
    combined_result.updated_themes = deduplicate_themes_by_title(combined_result.updated_themes)

    return combined_result


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
similarities. Look for themes with substantially overlapping content, even if their titles \
are not identical. For instance, 'Expert Instruction' and 'Expert Instructors' might \
cover similar content from different angles and should be considered for merging. Titles \
that are identical or have overlapping words are strong candidates for merging. \
Record your reasoning from this step.

Step 2:

Next, merge and refine themes based on your reasoning from the previous step:
- For each set of similar themes, merge them into one theme.
    - Combine their citations into one list.
    - Create a new, consolidated title that captures the essence of the merged theme.
    - Write a comprehensive description that encompasses all aspects of the themes being merged.
- For each unique theme that doesn't need merging, leave it as is.

Save the resulting updated themes from this step. Include all of the new \
merged ones and the unique ones that didn't need merging. MAKE SURE you \
keep ALL of the unique themes. You should not end up with more themes than you \
started with, given that you are only merging or keeping themes, not splitting themes. \
For every theme, include the 'theme_title' and 'description'. You should return an empty \
list for citations since those won't be needed for the next step.

Do your best. If you have done an outstanding job, you will get a $500 tip."""

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
