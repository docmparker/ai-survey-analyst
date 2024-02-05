# from instructor.function_calls import OpenAISchema
from .utils import OpenAISchema
from .single_input_task import InputModel, SurveyTaskProtocol, CommentModel, apply_task_with_logprobs, LLMConfig
from pydantic import Field, validate_arguments, model_validator
from typing import Any, Type, Literal, NamedTuple
from functools import partial
from . import batch_runner as br
import openai.types.chat.chat_completion
import numpy as np
import math


# Create the model - here we do it outside the class so it can also be used elsewhere if desired
# without instantiating the class
class SentimentAnalysisResult(OpenAISchema):
    """Store the sentiment and reasoning for a survey comment"""
    reasoning: str = Field("The comment had no content", description="The reasoning for the sentiment assignment")
    sentiment: Literal["positive", "neutral", "negative"] | None = Field(None, description="The sentiment of the comment")
    logprobs: openai.types.chat.chat_completion.ChoiceLogprobs | None = Field(None, description="The log probabilities for the sentiment assignment")
    sentiment_logprobs_: list[dict[str, Any]] | None = Field(None, description="The log probabilities for just the sentiment token")

    @property
    def sentiment_logprobs(self) -> list[dict[str, Any]] | None:
        """Returns the log probabilities and transformed (linear probs) 
        for just the sentiment token. 
        
        Note that this logic caches the result so that it is only computed once.
        Also note that this relies on the fact that 'positive', 'neutral', and 'negative'
        are all single tokens with the gpt-4 tokenizer.
        """

        if self.logprobs and not self.sentiment_logprobs_:
            def find_subsequence(seq, subseq):
                sub_len = len(subseq)
                for i in range(len(seq)):
                    if seq[i:i+sub_len] == subseq:
                        return i

            tokens = self.logprobs.model_dump().get('content')
            token_values = [token['token'] for token in tokens]
            subsequence = ['sent', 'iment', '":', ' "']
            index = find_subsequence(token_values, subsequence)

            if index:
                following_token = tokens[index + len(subsequence)]
            else:
                raise ValueError("Could not find the sentiment in the logprobs")

            self.sentiment_logprobs_ = [{'token': token['token'], 'logprob': token['logprob'], 'linear_prob': np.round(np.exp(token['logprob'])*100,2)}
                    for token in following_token['top_logprobs']]

        return self.sentiment_logprobs_   

    @property
    def classification_confidence(self) -> tuple[str, float]:
        """Calculate confidence based on top 2 distinct logprob diff within positive, negative, neutral rankings"""
        top_token, top_logprob = self.sentiment_logprobs[0]['token'], self.sentiment_logprobs[0]['logprob']

        next_logprob = None
        next_token = None
        for i, val in enumerate(self.sentiment_logprobs, start=1):
            if top_token.lower() != val['token'].lower():
                next_logprob = val['logprob']
                next_token = val['token']
                break

        class Confidence(NamedTuple):
            top_token: str
            next_token: str
            difference: float

        return Confidence(top_token, next_token, top_logprob - next_logprob if next_logprob is not None else math.inf)
    
    @property
    def top_logprob(self) -> float:
        """Returns the logprob for the top sentiment token"""
        return self.sentiment_logprobs[0]['logprob']


class SentimentAnalysis(SurveyTaskProtocol):
    """Class for sentiment analysis task"""
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel

    def prompt_messages(self, task_input: CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the sentiment analysis prompt"""

        delimiter = "####"
        # question = "What could be improved about the course?"

        system_message = f"""You are an assistant that determines the sentiment of \
student course feedback comments.  You respond only with a JSON object.

You will be provided with a comment from a student course feedback survey. \
The comment will be delimited by {delimiter} characters. \
Each original comment was in response to the question: "{self.question}". \
Your task is to determine the sentiment of the comment and provide your reasoning for the sentiment. \

Step 1: Reason through what the sentiment of the comment is and why.
- Include your reasoning in the "reasoning" field.

Step 2. Record the sentiment of the comment.
- The sentiment MUST be one of three categories: "positive", "neutral", or "negative".
- Include the sentiment in the "sentiment" field.

Do your best. I will tip you $500 if you do an excellent job."""


        user_message = f"""{delimiter}{task_input.comment}{delimiter}"""

        messages =  [  
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> Type[OpenAISchema]:
        """Returns the result class for sentiment analysis"""
        return SentimentAnalysisResult
    

@validate_arguments
async def classify_sentiment(*, comments: list[str | None], question: str) -> list[OpenAISchema]:
    """Classify the sentiment for each of a list of comments, based on a particular question 
    
    Returns a list of SentimentAnalysisResult objects
    """

    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    survey_task: SurveyTaskProtocol = SentimentAnalysis(question=question)
    sentiment_task = partial(apply_task_with_logprobs, 
                      get_prompt=survey_task.prompt_messages, 
                      result_class=survey_task.result_class,
                      llm_config=LLMConfig(logprobs=True, top_logprobs=3))
    sentiment_results = await br.process_tasks(comments_to_test, sentiment_task)

    return sentiment_results


def sort_by_confidence(comments: list[str], sentiment_results: list[SentimentAnalysisResult]) -> list[tuple[str, SentimentAnalysisResult]]:
    """Sort the comments and sentiment results by confidence, while keeping track of which comment goes with which result.
    The confidence is calculated as the difference between the top and next highest logprob for the sentiment token, with the 
    next highest logprob being the next highest distinct logprob ('positive' and 'Positive' are not distinct, for example).
    
    The results are sorted within each sentiment category, so that the top confidence for each sentiment category is at the top.
    """
    pairs = list(zip(comments, sentiment_results))

    # Sort the pairs based on the result
    pairs.sort(key=lambda pair: (pair[1].sentiment_logprobs[0]['token'], pair[1].classification_confidence.difference), reverse=True)

    return pairs

def sort_by_top_logprob(comments: list[str], sentiment_results: list[SentimentAnalysisResult]) -> list[tuple[str, SentimentAnalysisResult]]:
    """Sort the comments and sentiment results by top logprob, while keeping track of which comment goes with which result.
    
    The results are sorted within each sentiment category, so that the top logprob for each sentiment category is at the top.
    """
    
    pairs = list(zip(comments, sentiment_results))

    # Sort the pairs based on the result
    pairs.sort(key=lambda pair: (pair[1].sentiment_logprobs[0]['token'], pair[1].top_logprob), reverse=True)

    return pairs