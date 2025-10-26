# dspy.experimental.Citations

## dspy.experimental.Citations

```python
class Citations
```

Citations extracted from an LM response with source references.

This type represents citations returned by language models that support
citation extraction, particularly Anthropic's Citations API through LiteLLM.
Citations include the quoted text and source information.

Example:
    ```python
    import os
    import dspy
    from dspy.signatures import Signature
    from dspy.experimental import Citations, Document
    os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

    class AnswerWithSources(Signature):
        '''Answer questions using provided documents with citations.'''
        documents: list[Document] = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        citations: Citations = dspy.OutputField()

    # Create documents to provide as sources
    docs = [
        Document(
            data="The Earth orbits the Sun in an elliptical path.",
            title="Basic Astronomy Facts"
        ),
        Document(
            data="Water boils at 100°C at standard atmospheric pressure.",
            title="Physics Fundamentals",
            metadata={"author": "Dr. Smith", "year": 2023}
        )
    ]

    # Use with a model that supports citations like Claude
    lm = dspy.LM("anthropic/claude-opus-4-1-20250805")
    predictor = dspy.Predict(AnswerWithSources)
    result = predictor(documents=docs, question="What temperature does water boil?", lm=lm)

    for citation in result.citations.citations:
        print(citation.format())
    ```


### description

```python
def description(cls)
```

Description of the citations type for use in prompts.


### format

```python
def format(self)
```

Format citations as a list of dictionaries.


### from_dict_list

```python
def from_dict_list(cls, citations_dicts)
```

Convert a list of dictionaries to a Citations instance.

Args:
    citations_dicts: A list of dictionaries, where each dictionary should have 'cited_text' key
        and 'document_index', 'start_char_index', 'end_char_index' keys.

Returns:
    A Citations instance.

Example:
    ```python
    citations_dict = [
        {
            "cited_text": "The sky is blue",
            "document_index": 0,
            "document_title": "Weather Guide",
            "start_char_index": 0,
            "end_char_index": 15,
            "supported_text": "The sky was blue yesterday."
        }
    ]
    citations = Citations.from_dict_list(citations_dict)
    ```


### is_streamable

```python
def is_streamable(cls)
```

Whether the Citations type is streamable.


### parse_lm_response

```python
def parse_lm_response(cls, response)
```

Parse a LM response into Citations.

Args:
    response: A LM response that may contain citation data.

Returns:
    A Citations object if citation data is found, None otherwise.


### parse_stream_chunk

```python
def parse_stream_chunk(cls, chunk)
```

Parse a stream chunk into Citations.

Args:
    chunk: A stream chunk from the LM.

Returns:
    A Citations object if the chunk contains citation data, None otherwise.


### validate_input

```python
def validate_input(cls, data)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/types/citation.py` (lines 10–218)

