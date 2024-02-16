from .utils import OpenAISchema
from .models_common import InputModel, SurveyTaskProtocol, CommentModel, LLMConfig
from .single_input_task import apply_task_with_logprobs
from pydantic import Field, validate_arguments, computed_field, BaseModel, field_serializer
from typing import Any, Type, Literal, NamedTuple
from functools import partial, cached_property
from . import batch_runner as br
import openai.types.chat.chat_completion
import numpy as np
import math


class Confidence(BaseModel):
    top_token: str
    next_token: str | None
    difference: float = Field(allow_inf_nan=True)

    @field_serializer('difference')
    def serialize_dt(self, difference: float, _info):
        return "Infinity" if math.isinf(difference) else difference


class SentimentAnalysisResult(OpenAISchema):
    """Store the sentiment and reasoning for a survey comment"""
    reasoning: str = Field("The comment had no content", description="The reasoning for the sentiment assignment")
    sentiment: Literal["positive", "neutral", "negative"] | None = Field(None, description="The sentiment of the comment")
    logprobs: openai.types.chat.chat_completion.ChoiceLogprobs | None = Field(None, description="The log probabilities for the sentiment assignment")
    # sentiment_logprobs_: list[dict[str, Any]] | None = Field(None, description="The log probabilities for just the sentiment token")

    @computed_field
    @cached_property
    def sentiment_logprobs(self) -> list[dict[str, Any]] | None:
        """Returns the log probabilities and transformed (linear probs) 
        for just the sentiment token. 
        
        Note that this logic caches the result so that it is only computed once.
        Also note that this relies on the fact that 'positive', 'neutral', and 'negative'
        are all single tokens with the gpt-4 tokenizer.
        """

        if self.logprobs:
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

            return [{'token': token['token'], 'logprob': token['logprob'], 'linear_prob': np.round(np.exp(token['logprob'])*100,2)}
                    for token in following_token['top_logprobs']]
        else:
            return None

    @computed_field
    @cached_property
    def classification_confidence(self) -> Confidence | None:
        """Calculate confidence based on top 2 distinct logprob diff within positive, negative, neutral rankings"""

        if not self.sentiment_logprobs:
            # if the comment had no content, there will be no logprobs
            return None

        top_token, top_logprob = self.sentiment_logprobs[0]['token'], self.sentiment_logprobs[0]['logprob']

        next_logprob = None
        next_token = None
        for i, val in enumerate(self.sentiment_logprobs, start=1):
            # allow for 'positive' and 'Positive' and 'posit' and ' positive' to be considered the same
            if top_token.lower().strip()[:5] != val['token'].lower().strip()[:5]:
                next_logprob = val['logprob']
                next_token = val['token']
                break
        difference = (top_logprob - next_logprob) if next_logprob else math.inf

        return Confidence(top_token=top_token, 
                          next_token=next_token, 
                          difference=difference)
    
    @property
    def top_logprob(self) -> float:
        """Returns the logprob for the top sentiment token"""
        if not self.sentiment_logprobs:
            # if the comment had no content, there will be no logprobs
            raise ValueError("The comment had no content")

        return self.sentiment_logprobs[0]['logprob']

    @computed_field
    @cached_property
    def fine_grained_sentiment_category(self) -> str:
        """Return the sentiment category at a finer-grained level based on the logprobs.
        This basically keeps postive and negative but divides neutral into neutral-positive,
        neutral-negative, and neutral based on the next highest differing sentiment logprob.
        This is useful for color coding the sentiment categories, for example. 
        
        The returned categories are: positive, neutral, negative, neutral-positive, 
        neutral-negative, no_content.
        """

        if not self.sentiment_logprobs:
            return 'no_content'
        elif self.sentiment == 'positive':
            return 'positive'
        elif self.sentiment == 'negative':
            return 'negative'
        elif self.sentiment == 'neutral':
            next_token = self.classification_confidence.next_token or 'no_value'
            if next_token.lower().strip()[:5] == 'posit':
                return 'neutral-positive'
            elif next_token.lower().strip() == 'negative':
                return 'neutral-negative'
            else:
                return 'neutral'
        else:
            return 'no_content'


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
async def classify_sentiment(*, comments: list[str | float | None], question: str) -> list[OpenAISchema]:
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


def sort_by_confidence(comments: list[str | float | None], sentiment_results: list[SentimentAnalysisResult]) -> list[tuple[str, SentimentAnalysisResult]]:
    """Sort the comments and sentiment results by confidence, while keeping track of which comment goes with which result.
    The confidence is calculated as the difference between the top and next highest logprob for the sentiment token, with the 
    next highest logprob being the next highest distinct logprob ('positive' and 'Positive' are not distinct, for example).
    
    The results are sorted to largely go from most positive to most negative based on criteria as follows:
    - first sort by token (positive, neutral, negative)
    - then sort differently within each token
        - for positive, sort by difference, then by top logprob, reversed so goes from top difference to lowest, with logprob breaking ties
        - for negative, sort by difference, then by top logprob, so it goes from least negative to most negative, with logprob breaking ties
        - neutral is trickier, given that we may have neutral, neutral-negative, and neutral-positive, depending on what the next ranked
            different token from neutral is in the top_logprobs. We sort from most positive neutrals to most negative neutrals,
            with neutral neutrals being sorted by difference and then by top logprob.

    For ties (like ones that have math.inf as the difference), the sort is by the top logprob.
    """
    # ignore the comments that had no content
    pairs = [(comment, sentiment_result) for comment, sentiment_result in zip(comments, sentiment_results) if sentiment_result.sentiment_logprobs]

    # Sort the pairs based on the result
    # sort by token, then by difference, then by top logprob
    # pairs.sort(key=lambda pair: (pair[1].sentiment_logprobs[0]['token'], 
    #                              pair[1].classification_confidence.difference,
    #                              pair[1].top_logprob), 
    #                              reverse=True)

    # first sort by token
    def by_token(pair):
        token = pair[1].sentiment_logprobs[0]['token']
        return 1 if token == 'positive' else 2 if token == 'neutral' else 3

    def by_custom_criteria(pair):
        token = pair[1].sentiment_logprobs[0]['token']
        if token == 'positive':
            return (-pair[1].classification_confidence.difference, -pair[1].top_logprob)
        elif token == 'negative':
            return (pair[1].classification_confidence.difference, pair[1].top_logprob)
        elif token == 'neutral':
            next_token = pair[1].classification_confidence.next_token or 'no_value'
            if next_token.lower().strip()[:5] == 'posit':
                # if the next token is positive, then the highest difference and higher logprob is more neutral
                # so go from least neutral (more positive) to more neutral (less positive)
                return (1, pair[1].classification_confidence.difference, pair[1].top_logprob)
            elif next_token.lower().strip() == 'negative':
                # if the next token is negative, then the highest difference and higher logprob is more neutral
                # so go from more neutral (less negative) to least neutral (more negative)
                return (4, -pair[1].classification_confidence.difference, -pair[1].top_logprob)
            else:
                # the next token is neutral or no_value, so the order is not terribly important since
                # they are all presumably pretty neutral by the model's estimation
                return (2, pair[1].classification_confidence.difference, pair[1].top_logprob)
        else:
            raise ValueError(f"Unknown token: {token}")

    # relying on stable sort to keep the custom criteria sort within the token sort
    pairs.sort(key=by_custom_criteria)
    pairs.sort(key=by_token)

    return pairs

def sort_by_top_logprob(comments: list[str], sentiment_results: list[SentimentAnalysisResult]) -> list[tuple[str, SentimentAnalysisResult]]:
    """Sort the comments and sentiment results by top logprob, while keeping track of which comment goes with which result.
    
    The results are sorted within each sentiment category, so that the top logprob for each sentiment category is at the top.
    """
    
    # ignore the comments that had no content
    pairs = [(comment, sentiment_result) for comment, sentiment_result in zip(comments, sentiment_results) if sentiment_result.sentiment_logprobs]

    # Sort the pairs based on the result
    pairs.sort(key=lambda pair: (pair[1].sentiment_logprobs[0]['token'], pair[1].top_logprob), reverse=True)

    return pairs