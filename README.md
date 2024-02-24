# ai-survey-analyst

_Use LLMs to analyze your survey data_

AI-survey-analyst helps you answer common questions about your survey feedback data. Although it was designed with education feedback in mind (for example, all the student comments that come back from online or in-person course surveys), it could be adapted for user feedback of almost any type.

## Reasons to Use AI-survey-analyst

This package comes from my own personal experience of trying to use educational survey results to analyze and improve courses. Judging by eye gives an incomplete and potentially biased view of the feedback. When you have hundreds or thousands of comments, that's not even possible. Trying to rigorously code the comments manually can take significant time and expense. 

The high level questions that educators and other course designers/planners are often trying to investigate include:

1. What did students say about the course? (themes, summarization)
2. How did they feel about the course? (sentiment analysis)
3. What did they say about some aspect of interest? (extraction)
4. How did they feel about that aspect (focused sentiment analysis)
5. How many comments were there about different aspects? (multilabel classification)

This tool is designed to help you answer these questions and give you the information to make informed decisions based on feedback.

If you're interested in this sort of thing, have a look at the paper called [A Large Language Model Approach to Educational Survey Feedback Analysis](https://arxiv.org/pdf/2309.17447.pdf). 


## Example usage

Here is an example of deriving the main themes from a set of survey comments. 

```python
from survey_analysis.theme_derivation import derive_themes

best_parts_question = "What were the best parts of the course?"
best_parts_comments = example_survey['best_parts'].tolist() 
best_parts_themes = await derive_themes(comments=best_parts_comments, 
                                        question=best_parts_question) 

# let's see the titles of the derived themes
for theme in best_parts_themes.updated_themes:
    print(theme.theme_title)
```


## Installing

Download or git clone the repository. Then `cd` to the `ai-survey-analyst` directory.

Start your virtual environment of choice, for example via conda, so the dependencies don't interfere with your default environment. Then install the package with `pip`. The following makes an editable install in case you'd like to modify the code at some point, say for tweaking the prompts or adding a new survey task.

`pip install -e .`

If you want the dependencies needed to run the example notebooks (for example, Pandas, matplotlib, etc.), then install with the `nb` option (see below). There is a lot of good information in the example notebooks in terms of usage, so I'd highly recommend installing as follows with the optional notebook dependencies:

`pip install -e .[nb]`


## Example notebooks

Start with the `end_to_end_example.ipynb` for an overview and usage with various tasks. If you want to drill down further on any of the tasks, there are example notebooks devoted to each of the various survey tasks.


## License

The code is MIT license. 

One of the example datasets is a sample of 200 rows from a larger (1.45 million row) open [Coursera reviews dataset](https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera?resource=download). That dataset is licensed under GPL v2, but is not used in the survey_analysis code (only in the example notebooks), so does not affect the MIT license. 
