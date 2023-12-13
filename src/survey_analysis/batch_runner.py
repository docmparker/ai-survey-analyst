from instructor.function_calls import OpenAISchema, Mode
from typing import Callable
import time
import asyncio

RunTask = Callable[[str], OpenAISchema]

async def process_tasks(comments: list[str],
                                      run_task: RunTask,
                                      batch_size: int=100, 
                                      batch_sleep_interval: int=30) -> list[OpenAISchema]:
    """Takes a list of comments and processes them in parallel, returning a list of 
    pydantic model responses. This is done in batches, with the batches having their 
    model calls processed in parallel.
    """

    print(f"processing {len(comments)} comments in batches of {batch_size}")
    print(f"sleeping for {batch_sleep_interval} seconds between batches")
    response_list: list[OpenAISchema] = []
    for i in range(0, len(comments), batch_size):
        comment_batch = comments[i:i+batch_size]

        print(f"starting {i} to {i+batch_size}")
        start_time = time.time()

        # responses = [1]*len(comment_batch)
        responses = await asyncio.gather(*[run_task(comment) for comment in comment_batch])
        response_list.extend(responses)
        print(f"completed {i} to {i+batch_size}")
        # logging.info(f"completed {i} to {i+batch_size}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"elapsed time: {elapsed_time}")

        if i < len(comments) - batch_size:
            time_to_next_minute = batch_sleep_interval - (elapsed_time % batch_sleep_interval)
            print(f"sleeping for {time_to_next_minute} seconds")
            time.sleep(time_to_next_minute)

    return response_list
