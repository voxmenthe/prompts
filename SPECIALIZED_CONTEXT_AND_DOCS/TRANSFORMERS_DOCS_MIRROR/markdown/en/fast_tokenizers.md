# Tokenizers

Tokenizers convert text into an array of numbers known as tensors, the inputs to a text model. There are several tokenizer algorithms, but they all share the same purpose. Split text into smaller words or subwords (tokens) according to some rules, and convert them into numbers (input ids). A Transformers tokenizer also returns an attention mask to indicate which tokens should be attended to.

Learn about the most popular tokenization algorithms on the [Summary of the tokenizers](./tokenizer_summary) doc.

Call [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained) to load a tokenizer and its configuration from the Hugging Face [Hub](https://hf.co) or a local directory. The pretrained tokenizer is saved in a [tokenizer.model](https://huggingface.co/google/gemma-2-2b/blob/main/tokenizer.model) file with all its associated vocabulary files.

Pass a string of text to the tokenizer to return the input ids and attention mask, and set the framework tensor type to return with the `return_tensors` parameter.


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("We are very happy to show you the 🤗 Transformers library", return_tensors="pt")
{'input_ids': tensor([[     2,   1734,    708,   1508,   4915,    577,   1500,    692,    573,
         156808, 128149,   9581, 235265]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

Whichever tokenizer you use, make sure the tokenizer vocabulary is the same as the pretrained models tokenizer vocabulary. This is especially important if you’re using a custom tokenizer with a different vocabulary from the pretrained models tokenizer.

This guide provides a brief overview of the tokenizer classes and how to preprocess text with it.

## Tokenizer classes

All tokenizers inherit from a [PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) class that provides common methods for all tokenizers like [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained) and [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). There are two main tokenizer classes that build on top of the base class.

* [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) is a Python implementation, for example [LlamaTokenizer](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizer).
* [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) is a fast Rust-based implementation from the [Tokenizers](https://hf.co/docs/tokenizers/index) library, for example [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast).

There are two ways you can load a tokenizer, with [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) or a model-specific tokenizer.

AutoTokenizer

model-specific tokenizer

The [AutoClass](./model_doc/auto) API is a fast and easy way to load a tokenizer without needing to know whether a Python or Rust-based implementation is available. By default, [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) tries to load a fast tokenizer if it’s available, otherwise, it loads the Python implementation.

Use [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained) to load a tokenizer.


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("We are very happy to show you the 🤗 Transformers library.", return_tensors="pt")
{'input_ids': tensor([[     2,   1734,    708,   1508,   4915,    577,   1500,    692,    573,
         156808, 128149,   9581, 235265]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

Load your own tokenizer by passing its vocabulary file to [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained).


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./model_directory/my_vocab_file.txt")
```

## Multimodal tokenizers

In addition to text tokens, multimodal tokenizers also holds tokens from other modalities as a part of its attributes for easy access.

To add these special tokens to a tokenizer, pass them as a dictionary to the `extra_special_tokens` parameter in [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained). The example below adds the `image_token` to a vision-language model.

Save the tokenizer so you can reuse it with direct access to the `image_token`, `boi_token`, and `eoi_token`.


```
vision_tokenizer = AutoTokenizer.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    extra_special_tokens={"image_token": "<image>", "boi_token": "<image_start>", "eoi_token": "<image_end>"}
)
print(vision_tokenizer.image_token, vision_tokenizer.image_token_id)
("<image>", 32000)

vision_tokenizer.save_pretrained("./path/to/tokenizer")
```

## Fast tokenizers

[PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) or *fast tokenizers* are Rust-based tokenizers from the [Tokenizers](https://hf.co/docs/tokenizers) library. It is significantly faster at batched tokenization and provides additional alignment methods compared to the Python-based tokenizers.

[AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer) automatically loads a fast tokenizer if it’s supported. Otherwise, you need to explicitly load the fast tokenizer.

This section will show you how to train a fast tokenizer and reuse it in Transformers.

To train a Byte-Pair Encoding (BPE) tokenizer, create a `Tokenizer` and `BpeTrainer` class and define the unknown token and special tokens.


```
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
```

Split the tokens on `Whitespace` to create tokens that don’t overlap with each other.


```
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

Call `train` on the text files and trainer to start training.


```
files = [...]
tokenizer.train(files, trainer)
```

Use `save` to save the tokenizers configuration and vocabulary to a JSON file.


```
tokenizer.save("tokenizer.json")
```

Now you can load and reuse the tokenizer object in Transformers by passing it to the `tokenizer_object` parameter in [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast).


```
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

To load a saved tokenizer from its JSON file, pass the file path to the `tokenizer_file` parameter in [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast).


```
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

## tiktoken

[tiktoken](https://github.com/openai/tiktoken) is a [byte-pair encoding (BPE)](./tokenizer_summary#byte-pair-encoding-bpe) tokenizer by OpenAI. It includes several tokenization schemes or encodings for how text should be tokenized.

There are currently two models trained and released with tiktoken, GPT2 and Llama3. Transformers supports models with a [tokenizer.model](https://hf.co/meta-llama/Meta-Llama-3-8B/blob/main/original/tokenizer.model) tiktoken file. The tiktoken file is automatically converted into Transformers Rust-based [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast).

Add the `subfolder` parameter to [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to specify where the `tokenizer.model` tiktoken file is located.


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", subfolder="original")
```

### Create a tiktoken tokenizer

The tiktoken `tokenizer.model` file contains no information about additional tokens or pattern strings. If these are important, convert the tokenizer to `tokenizer.json` (the appropriate format for [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast)).

Generate the tiktoken `tokenizer.model` file with the [tiktoken.get\_encoding](https://github.com/openai/tiktoken/blob/63527649963def8c759b0f91f2eb69a40934e468/tiktoken/registry.py#L63) function, and convert it to `tokenizer.json` with [convert\_tiktoken\_to\_fast](https://github.com/huggingface/transformers/blob/99e0ab6ed888136ea4877c6d8ab03690a1478363/src/transformers/integrations/tiktoken.py#L8).


```
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding

# Load your custom encoding or the one provided by OpenAI
encoding = get_encoding("gpt2")
convert_tiktoken_to_fast(encoding, "config/save/dir")
```

The resulting `tokenizer.json` file is saved to the specified directory and loaded with [from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained).


```
tokenizer = PreTrainedTokenizerFast.from_pretrained("config/save/dir")
```

## Preprocess

A Transformers model expects the input to be a PyTorch or NumPy tensor. A tokenizers job is to preprocess text into those tensors. Specify the framework tensor type to return with the `return_tensors` parameter.


```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("We are very happy to show you the 🤗 Transformers library.", return_tensors="pt")
{'input_ids': tensor([[     2,   1734,    708,   1508,   4915,    577,   1500,    692,    573,
         156808, 128149,   9581, 235265]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

The tokenization process of converting text into input ids is completed in two steps.

1. tokenize

2. convert tokens to ids

3. decode ids to text

In the first step, a string of text is split into tokens by the [tokenize()](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.tokenize) function. How the text is split depends on the tokenization algorithm.


```
tokens = tokenizer.tokenize("We are very happy to show you the 🤗 Transformers library")
print(tokens)
['We', '▁are', '▁very', '▁happy', '▁to', '▁show', '▁you', '▁the', '▁🤗', '▁Transformers', '▁library']
```

Gemma uses a [SentencePiece](./tokenizer_summary#sentencepiece) tokenizer which replaces spaces with an underscore `_`.

Visualize how different tokenizers work in the [Tokenizer Playground](https://xenova-the-tokenizer-playground.static.hf.space).

### Special tokens

Special tokens provide the model with some additional information about the text.

For example, if you compare the tokens obtained from passing text directly to the tokenizer and from [convert\_tokens\_to\_ids()](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.convert_tokens_to_ids), you’ll notice some additional tokens are added.


```
model_inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
[2, 1734, 708, 1508, 4915, 577, 1500, 692, 573, 156808, 128149, 9581]
tokenizer.convert_tokens_to_ids(tokens)
[1734, 708, 1508, 4915, 577, 1500, 692, 573, 156808, 128149, 9581]
```

When you [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode) the ids, you’ll see `<bos>` at the beginning of the string. This is used to indicate the beginning of a sentence to the model.


```
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
'<bos>We are very happy to show you the 🤗 Transformers library.'
'We are very happy to show you the 🤗 Transformers library'
```

Not all models need special tokens, but if they do, a tokenizer automatically adds them.

### Batch tokenization

It is faster and more efficient to preprocess *batches* of text instead of a single sentence at a time. Fast tokenizers are especially good at parallelizing tokenization.

Pass a list of string text to the tokenizer.


```
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences, return_tensors="pt")
print(encoded_inputs)
{
 'input_ids':
    [[2, 1860, 1212, 1105, 2257, 14457, 235336],
     [2, 4454, 235303, 235251, 1742, 693, 9242, 1105, 2257, 14457, 235269, 48782, 235265],
     [2, 1841, 1105, 29754, 37453, 235336]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1]]
}
```

### Padding

Learn about additional padding strategies in the [Padding and truncation](./pad_truncation) guide.

In the output above, the `input_ids` have different lengths. This is an issue because Transformers expects them to have the same lengths so it can pack them into a batch. Sequences with uneven lengths can’t be batched.

Padding adds a special *padding token* to ensure all sequences have the same length. Set `padding=True` to pad the sequences to the longest sequence length in the batch.


```
encoded_inputs = tokenizer(batch_sentences, padding=True, return_tensors="pt")
print(encoded_inputs)
```

The tokenizer added the special padding token `0` to the left side (*left padding*) because Gemma and LLMs in general are not trained to continue generation from a padding token.

### Truncation

Learn about additional truncation strategies in the [Padding and truncation](./pad_truncation) guide.

Models are only able to process sequences up to a certain length. If you try to process a sequence longer than a model can handle, it crashes.

Truncation removes tokens from a sequence to ensure it doesn’t exceed the maximum length. Set `truncation=True` to truncate a sequence to the maximum length accepted by the model. You can also set the maximum length yourself with the `max_length` parameter.


```
encoded_inputs = tokenizer(batch_sentences, max_length=8, truncation=True, return_tensors="pt")
print(encoded_inputs)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/fast_tokenizers.md)
