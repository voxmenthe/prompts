# dspy.streaming.StreamListener

## dspy.streaming.StreamListener

```python
class StreamListener(signature_field_name, predict=None, predict_name=None, allow_reuse=False)
```

Class that listens to the stream to capture the streeaming of a specific output field of a predictor.


### flush

```python
def flush(self)
```

Flush all tokens in the field end queue.

This method is called to flush out the last a few tokens when the stream is ended. These tokens
are in the buffer because we don't directly yield the tokens received by the stream listener
with the purpose to not yield the end_identifier tokens, e.g., "[[ ## ... ## ]]" for ChatAdapter.


### receive

```python
def receive(self, chunk)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/streaming/streaming_listener.py` (lines 22–361)

