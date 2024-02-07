# from instructor.function_calls import OpenAISchema
from .utils import OpenAISchema
from .single_input_task import InputModel, SurveyTaskProtocol, CommentModel, CommentBatch, apply_task
from pydantic import Field, validate_arguments
from typing import Type
from functools import partial
from . import batch_runner as br


class SummarizationResult(OpenAISchema):
    """Store the results of summarizing a group of survey comments"""
    summary: str = Field("The comments had no content", description="The summary of the comments")


class Summarization(SurveyTaskProtocol):
    """Class for comment summarization task"""
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentBatch

    def prompt_messages(self, task_input: CommentBatch) -> list[dict[str, str]]:
        """Creates the messages for the summarization prompt"""

        delimiter = "####"
        # question = "What could be improved about the course?"

        system_message = f"""You are an assistant that summarizes the themes of \
a set of student course feedback comments.  You respond only with a JSON object.

You will be provided with a group of comments from a student course feedback survey. \
Each comment will be delimited by {delimiter} characters. \
Each original comment was in response to the question: "{self.question}". \
Your goal is to summarize the themes of feedback shared by two or more students (each comment \
corresponds to a single student). A theme is a succinct (two sentences or less) \
description of such common feedback. \
Your summary should describe the all of the unique \
themes (in other words, all of the themes that are distinct \
from each other in terms of the feedback they represent) and comprehensive (in other words, \
encompass ALL feedback that is expressed by two or more students). If a particular subject of \
feedback is only shared by one student, it should NOT be included in the summary.

Do your best. I will tip you $500 if you do an excellent job."""

        comment_list_del = "\n".join([f"{delimiter}{comment.comment}{delimiter}" 
                                      for comment in task_input.comments 
                                      if not comment.is_empty()])
        
        user_message = f"""{comment_list_del}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for summarization"""
        return SummarizationResult
    

@validate_arguments
async def summarize_comments(*, comments: list[str | float | None], question: str) -> OpenAISchema:
    """Summarize the themes of a group of comments, based on a particular question
    
    Returns a SummarizationResult object
    """
    comment_list = [CommentModel(comment=comment) for comment in comments]
    comments_to_test: CommentBatch = CommentBatch(comments=comment_list)
    survey_task: SurveyTaskProtocol = Summarization(question=question)
    summarization_task = partial(apply_task, 
                      get_prompt=survey_task.prompt_messages, 
                      result_class=survey_task.result_class)
    summarization_result = await summarization_task(comments_to_test)

    return summarization_result