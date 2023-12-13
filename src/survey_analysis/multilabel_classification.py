from enum import Enum
from pydantic import Field
from pydantic.main import create_model
from instructor.function_calls import OpenAISchema
import yaml
from .single_comment_task import SurveyTaskProtocol

# load the tags as a list of dicts, each with a 'topic' and 'description'
# in the yaml, these are all under the root of 'tags'
# tags_predefined = yaml.safe_load(open('../data/tags_8.yaml', 'r'))['tags']

with open('../data/tags_8.yaml', 'r') as file:
    data = yaml.safe_load(file)

tags_list = data['tags']

# Model classes
class CategoryValue(Enum):
    ZERO = 0
    ONE = 1

# assume all categories are 0 if there is no content in the comment (an alternative would be to set 'other' to 1 if there is such
# a category and there is no content in the comment). Defaults are set to CategoryValue.ZERO.
tag_fields = {'_'.join(tag['topic'].split()): (CategoryValue, CategoryValue.ZERO) for tag in tags_list}
CategoriesModel = create_model('Categories', **tag_fields, __base__=OpenAISchema)

# I am deciding to only include the descriptions of the categories spelled out in the system message to the 
# model but not to repeate those in the field descriptions for the schema.

# Create the model
class MultilabelClassificationResult(OpenAISchema):
    """Store the multilabel classification and reasoning of a comment"""
    reasoning: str = Field("The comment had no content", description="The reasoning for the classification")
    categories: CategoriesModel = Field(CategoriesModel(), description="The categories that the comment belongs to")


class MultiLabelClassification(SurveyTaskProtocol):
    """Class for multilabel classification"""
    def __init__(self, tags_list: list[dict[str, str]]):
        self.tags_list = tags_list

    def prompt_messages(self, comment: str) -> list[dict[str, str]]:
        """Creates the messages for the multilabel classification prompt

        tags_for_prompt is a list of tags that will be used in the prompt, each
        a dict with a 'topic' and 'description' key.
        """

        def format_topic_name(topic: str):
            """Format a topic name for use in the prompt"""
            return topic.replace(' ', '_').lower()
        
        def format_tag(tag: dict):
            """Format a tag for use in the prompt"""
            return f"Topic: {format_topic_name(tag['topic'])}\nDescription: {tag['description']}"
        
        tags_for_prompt = '\n\n'.join([format_tag(tag) for tag in self.tags_list])

        system_message = f"""You are an assistant that classifies student course \
feedback comments.  You respond only with a JSON object.

You will be provided with a comment from a student course feedback survey. \
Categorize the comment with as many of the following categories as apply:

{tags_for_prompt}

Think step-by-step to arrive at a correct classification. Include your reasoning \
behind every assigned category in the output."""

        user_message = f"""{comment}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> MultilabelClassificationResult:
        """Returns the result class for multilabel classification"""
        return MultilabelClassificationResult
    