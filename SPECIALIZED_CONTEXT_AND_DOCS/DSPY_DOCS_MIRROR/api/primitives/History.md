# dspy.History

## dspy.History

```python
class History
```

Class representing the conversation history.

The conversation history is a list of messages, each message entity should have keys from the associated signature.
For example, if you have the following signature:

```
class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    history: dspy.History = dspy.InputField()
    answer: str = dspy.OutputField()
```

Then the history should be a list of dictionaries with keys "question" and "answer".

Example:
    ```
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    predict = dspy.Predict(MySignature)
    outputs = predict(question="What is the capital of France?", history=history)
    ```

Example of capturing the conversation history:
    ```
    import dspy

    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    predict = dspy.Predict(MySignature)
    outputs = predict(question="What is the capital of France?")
    history = dspy.History(messages=[{"question": "What is the capital of France?", **outputs}])
    outputs_with_history = predict(question="Are you sure?", history=history)
    ```

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/types/history.py` (lines 6–68)

