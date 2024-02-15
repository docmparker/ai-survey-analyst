from enum import Enum
from pydantic import Field, ConfigDict, validate_arguments
from pydantic.main import create_model
from functools import partial
from .utils import OpenAISchema
import yaml
from .models_common import SurveyTaskProtocol, InputModel, CommentModel
from .single_input_task import apply_task
from . import batch_runner as br
from typing import Type

# load the tags as a list of dicts, each with a 'topic' and 'description'
# in the yaml, these are all under the root of 'tags'
# It is loaded here for the default set of tags we developed for course feedback
with open('../data/tags_8.yaml', 'r') as file:
    data = yaml.safe_load(file)

default_tags_list: list[dict[str, str]] = data['tags']


class MultiLabelClassification(SurveyTaskProtocol):
    """Class for multilabel classification"""
    def __init__(self, tags_list: list[dict[str, str]]):
        """Initialize the multilabel classification task with a list of tags, 
        each a dict with a 'topic' and 'description' key"""
        self.tags_list = tags_list
        self._result_class = None
        self._tags_for_prompt = None

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel
    
    @property
    def tags_for_prompt(self) -> str:
        """tags_for_prompt is a list of tags that will be used in the prompt, each
        a dict with a 'topic' and 'description' key.
        """

        def format_topic_name(topic: str):
            """Format a topic name for use in the prompt"""
            return topic.replace(' ', '_').lower()
        
        def format_tag(tag: dict):
            """Format a tag for use in the prompt"""
            return f"Topic: {format_topic_name(tag['topic'])}\nDescription: {tag['description']}"

        if not self._tags_for_prompt: 
            tags_for_prompt = '\n\n'.join([format_tag(tag) for tag in self.tags_list])
            self._tags_for_prompt = tags_for_prompt

        return self._tags_for_prompt


    def prompt_messages(self, task_input:CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the multilabel classification prompt"""

        delimiter = "####"

        system_message = f"""You are an assistant that classifies student course \
feedback comments.  You respond only with JSON output.

You will be provided with a comment from a student course feedback survey. \
The comment will be delimited by {delimiter} characters. \
The goal is to categorize the comment with as many of the following categories as apply:

{self._tags_for_prompt}

Step 1: Reason through what categories you will choose and why.
- Include your reasoning in the "reasoning" field.

Step 2. Record the categories you choose.
- Record the categories you choose in the "categories" field.
- If more than one category applies, include all that apply.

When you are done categorizing the comment, return your reasoning and categories \
in the JSON output. Do your best. I will tip you $500 if you do an excellent job."""

        user_message = f"""{delimiter}{task_input.comment}{delimiter}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for multilabel classification, dynamically creating it if necessary"""

        if not self._result_class:
            # only need to dynamically create the model once
            class CategoryValue(Enum):
                ZERO = 0
                ONE = 1

            aliases = {'_'.join(tag['topic'].split()): tag['topic'] for tag in self.tags_list}
            def tag_alias_generator(name: str) -> str:
                """Generate aliases based on what was passed with the tags"""
                return aliases[name]

            tag_fields = {'_'.join(tag['topic'].split()): (CategoryValue, CategoryValue.ZERO) for tag in self.tags_list}
            # CategoriesModel = create_model('Categories', **tag_fields, __base__=OpenAISchema)
            # note that the openaischema uses the aliases to create the json schema for the model
            CategoriesModel = create_model('Categories', **tag_fields, __config__=ConfigDict(alias_generator=tag_alias_generator, __base__=OpenAISchema))

            # I am deciding to only include the descriptions of the categories spelled out in the system message to the 
            # model but not to repeat those in the field descriptions for the schema.

            # Create the model
            class MultilabelClassificationResult(OpenAISchema):
                """Store the multilabel classification and reasoning of a comment"""
                reasoning: str = Field("The comment had no content", description="The reasoning for the classification")
                categories: CategoriesModel = Field(CategoriesModel(), description="The categories that the comment belongs to")

            self._result_class = MultilabelClassificationResult

        return self._result_class


@validate_arguments
async def multilabel_classify(*, comments: list[str | float | None], tags_list: list[dict[str, str]] | None = None) -> OpenAISchema:
    """Multilabel classify a list of comments, based on a list of categories (tags)
    
    Returns a list of ExcerptExtractionResult objects
    """

    if not tags_list:
        tags_list = default_tags_list

    survey_task: SurveyTaskProtocol = MultiLabelClassification(tags_list=tags_list)
    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    mlc_task = partial(apply_task, get_prompt=survey_task.prompt_messages, result_class=survey_task.result_class)
    classifications = await br.process_tasks(comments_to_test, mlc_task)

    return classifications