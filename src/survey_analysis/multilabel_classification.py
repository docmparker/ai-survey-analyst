from enum import Enum
from pydantic import Field, BaseModel
from pydantic.main import create_model
from instructor.function_calls import OpenAISchema
import yaml
from .single_input_task import SurveyTaskProtocol, InputModel, CommentModel
from typing import Type
from .utils import comment_has_content

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

        system_message = f"""You are an assistant that classifies student course \
feedback comments.  You respond only with a JSON object.

You will be provided with a comment from a student course feedback survey. \
Categorize the comment with as many of the following categories as apply:

{self._tags_for_prompt}

Think step-by-step to arrive at a correct classification. Include your reasoning \
behind every assigned category in the output."""

        user_message = f"""{task_input.comment}"""

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

            tag_fields = {'_'.join(tag['topic'].split()): (CategoryValue, CategoryValue.ZERO) for tag in self.tags_list}
            CategoriesModel = create_model('Categories', **tag_fields, __base__=OpenAISchema)

            # I am deciding to only include the descriptions of the categories spelled out in the system message to the 
            # model but not to repeat those in the field descriptions for the schema.

            # Create the model
            class MultilabelClassificationResult(OpenAISchema):
                """Store the multilabel classification and reasoning of a comment"""
                reasoning: str = Field("The comment had no content", description="The reasoning for the classification")
                categories: CategoriesModel = Field(CategoriesModel(), description="The categories that the comment belongs to")

            self._result_class = MultilabelClassificationResult

        return self._result_class
    