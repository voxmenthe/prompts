# dspy.Embedder

## dspy.Embedder

```python
class Embedder(model, batch_size=200, caching=True, **kwargs)
```

DSPy embedding class.

The class for computing embeddings for text inputs. This class provides a unified interface for both:

1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
2. Custom embedding functions that you provide

For hosted models, simply pass the model name as a string (e.g., "openai/text-embedding-3-small"). The class will use
litellm to handle the API calls and caching.

For custom embedding models, pass a callable function that:
- Takes a list of strings as input.
- Returns embeddings as either:
    - A 2D numpy array of float32 values
    - A 2D list of float32 values
- Each row should represent one embedding vector

Args:
    model: The embedding model to use. This can be either a string (representing the name of the hosted embedding
        model, must be an embedding model supported by litellm) or a callable that represents a custom embedding
        model.
    batch_size (int, optional): The default batch size for processing inputs in batches. Defaults to 200.
    caching (bool, optional): Whether to cache the embedding response when using a hosted model. Defaults to True.
    **kwargs: Additional default keyword arguments to pass to the embedding model.

Examples:
    Example 1: Using a hosted model.

    ```python
    import dspy

    embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
    embeddings = embedder(["hello", "world"])

    assert embeddings.shape == (2, 1536)
    ```

    Example 2: Using any local embedding model, e.g. from https://huggingface.co/models?library=sentence-transformers.

    ```python
    # pip install sentence_transformers
    import dspy
    from sentence_transformers import SentenceTransformer

    # Load an extremely efficient local model for retrieval
    model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

    embedder = dspy.Embedder(model.encode)
    embeddings = embedder(["hello", "world"], batch_size=1)

    assert embeddings.shape == (2, 1024)
    ```

    Example 3: Using a custom function.

    ```python
    import dspy
    import numpy as np

    def my_embedder(texts):
        return np.random.rand(len(texts), 10)

    embedder = dspy.Embedder(my_embedder)
    embeddings = embedder(["hello", "world"], batch_size=1)

    assert embeddings.shape == (2, 10)
    ```


### __call__

```python
def __call__(self, inputs, batch_size=None, caching=None, **kwargs)
```

Compute embeddings for the given inputs.

Args:
    inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
    batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set
        during initialization.
    caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None,
        defaults to the caching setting from initialization.
    kwargs: Additional keyword arguments to pass to the embedding model. These will override the default
        kwargs provided during initialization.

Returns:
    numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
    If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.


### acall

```python
async def acall(self, inputs, batch_size=None, caching=None, **kwargs)
```
Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/clients/embedding.py` (lines 9–147)

