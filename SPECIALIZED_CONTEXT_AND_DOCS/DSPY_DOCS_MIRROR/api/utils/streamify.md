# dspy.streamify

## dspy.streamify

```python
def streamify(program, status_message_provider=None, stream_listeners=None, include_final_prediction_in_output_stream=True, is_async_program=False, async_streaming=True)
```

Wrap a DSPy program so that it streams its outputs incrementally, rather than returning them
all at once. It also provides status messages to the user to indicate the progress of the program, and users
can implement their own status message provider to customize the status messages and what module to generate
status messages for.

Args:
    program: The DSPy program to wrap with streaming functionality.
    status_message_provider: A custom status message generator to use instead of the default one. Users can
        implement their own status message generator to customize the status messages and what module to generate
        status messages for.
    stream_listeners: A list of stream listeners to capture the streaming output of specific fields of sub predicts
        in the program. When provided, only the target fields in the target predict will be streamed to the user.
    include_final_prediction_in_output_stream: Whether to include the final prediction in the output stream, only
        useful when `stream_listeners` is provided. If `False`, the final prediction will not be included in the
        output stream. When the program hit cache, or no listeners captured anything, the final prediction will
        still be included in the output stream even if this is `False`.
    is_async_program: Whether the program is async. If `False`, the program will be wrapped with `asyncify`,
        otherwise the program will be called with `acall`.
    async_streaming: Whether to return an async generator or a sync generator. If `False`, the streaming will be
        converted to a sync generator.

Returns:
    A function that takes the same arguments as the original program, but returns an async
        generator that yields the program's outputs incrementally.

Example:

```python
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
# Create the program and wrap it with streaming functionality
program = dspy.streamify(dspy.Predict("q->a"))

# Use the program with streaming output
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

Example with custom status message provider:
```python
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyStatusMessageProvider(StatusMessageProvider):
    def module_start_status_message(self, instance, inputs):
        return f"Predicting..."

    def tool_end_status_message(self, outputs):
        return f"Tool calling finished with output: {outputs}!"

# Create the program and wrap it with streaming functionality
program = dspy.streamify(dspy.Predict("q->a"), status_message_provider=MyStatusMessageProvider())

# Use the program with streaming output
async def use_streaming():
    output = program(q="Why did a chicken cross the kitchen?")
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

Example with stream listeners:

```python
import asyncio
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False))

# Create the program and wrap it with streaming functionality
predict = dspy.Predict("question->answer, reasoning")
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
stream_predict = dspy.streamify(predict, stream_listeners=stream_listeners)

async def use_streaming():
    output = stream_predict(
        question="why did a chicken cross the kitchen?",
        include_final_prediction_in_output_stream=False,
    )
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value

output = asyncio.run(use_streaming())
print(output)
```

You should see the streaming chunks (in the format of `dspy.streaming.StreamResponse`) in the console output.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/streaming/streamify.py` (lines 27–220)

