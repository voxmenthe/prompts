# BertGeneration

[BertGeneration](https://huggingface.co/papers/1907.12461) leverages pretrained BERT checkpoints for sequence-to-sequence tasks with the [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) architecture. BertGeneration adapts the `BERT` for generative tasks.

You can find all the original BERT checkpoints under the [BERT](https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc) collection.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the BertGeneration models in the right sidebar for more examples of how to apply BertGeneration to different sequence generation tasks.

The example below demonstrates how to use BertGeneration with [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) for sequence-to-sequence tasks.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="google/roberta2roberta_L-24_discofuse",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create energy through ")
```

```python
import torch
from transformers import EncoderDecoderModel, AutoTokenizer

model = EncoderDecoderModel.from_pretrained("google/roberta2roberta_L-24_discofuse", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer(
    "Plants create energy through ", add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

```bash
echo -e "Plants create energy through " | transformers run --task text2text-generation --model "google/roberta2roberta_L-24_discofuse" --device 0
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [BitsAndBytesConfig](../quantizationbitsandbytes) to quantize the weights to 4-bit.

```python
import torch
from transformers import EncoderDecoderModel, AutoTokenizer, BitsAndBytesConfig

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = EncoderDecoderModel.from_pretrained(
    "google/roberta2roberta_L-24_discofuse",
    quantization_config=quantization_config,
    dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_discofuse")

input_ids = tokenizer(
    "Plants create energy through ", add_special_tokens=False, return_tensors="pt"
).input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

## Notes

- [BertGenerationEncoder](/docs/transformers/main/en/model_doc/bert-generation#transformers.BertGenerationEncoder) and [BertGenerationDecoder](/docs/transformers/main/en/model_doc/bert-generation#transformers.BertGenerationDecoder) should be used in combination with [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) for sequence-to-sequence tasks.

   ```python
   from transformers import BertGenerationEncoder, BertGenerationDecoder, BertTokenizer, EncoderDecoderModel
   
   # leverage checkpoints for Bert2Bert model
   # use BERT's cls token as BOS token and sep token as EOS token
   encoder = BertGenerationEncoder.from_pretrained("google-bert/bert-large-uncased", bos_token_id=101, eos_token_id=102)
   # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
   decoder = BertGenerationDecoder.from_pretrained(
       "google-bert/bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
   )
   bert2bert = EncoderDecoderModel(encoder=encoder, decoder=decoder)

   # create tokenizer
   tokenizer = BertTokenizer.from_pretrained("google-bert/bert-large-uncased")

   input_ids = tokenizer(
       "This is a long article to summarize", add_special_tokens=False, return_tensors="pt"
   ).input_ids
   labels = tokenizer("This is a short summary", return_tensors="pt").input_ids

   # train
   loss = bert2bert(input_ids=input_ids, decoder_input_ids=labels, labels=labels).loss
   loss.backward()
   ```

- For summarization, sentence splitting, sentence fusion and translation, no special tokens are required for the input.
- No EOS token should be added to the end of the input for most generation tasks.

## BertGenerationConfig[[transformers.BertGenerationConfig]]

#### transformers.BertGenerationConfig[[transformers.BertGenerationConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert_generation/configuration_bert_generation.py#L20)

This is the configuration class to store the configuration of a `BertGenerationPreTrainedModel`. It is used to
instantiate a BertGeneration model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the BertGeneration
[google/bert_for_seq_generation_L-24_bbc_encoder](https://huggingface.co/google/bert_for_seq_generation_L-24_bbc_encoder)
architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import BertGenerationConfig, BertGenerationEncoder

>>> # Initializing a BertGeneration config
>>> configuration = BertGenerationConfig()

>>> # Initializing a model (with random weights) from the config
>>> model = BertGenerationEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50358) : Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `BertGeneration`.

hidden_size (`int`, *optional*, defaults to 1024) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 24) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often called feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

hidden_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to 512) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

pad_token_id (`int`, *optional*, defaults to 0) : Padding token id.

bos_token_id (`int`, *optional*, defaults to 2) : Beginning of stream token id.

eos_token_id (`int`, *optional*, defaults to 1) : End of stream token id.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

## BertGenerationTokenizer[[transformers.BertGenerationTokenizer]]

#### transformers.BertGenerationTokenizer[[transformers.BertGenerationTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert_generation/tokenization_bert_generation.py#L30)

Construct a BertGeneration tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

save_vocabularytransformers.BertGenerationTokenizer.save_vocabularyhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_sentencepiece.py#L238[{"name": "save_directory", "val": ": str"}, {"name": "filename_prefix", "val": ": typing.Optional[str] = None"}]- **save_directory** (`str`) --
  The directory in which to save the vocabulary.
- **filename_prefix** (`str`, *optional*) --
  An optional prefix to add to the named of the saved files.0`tuple(str)`Paths to the files saved.

Save the sentencepiece vocabulary (copy original file) to a directory.

**Parameters:**

vocab_file (`str`) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.

bos_token (`str`, *optional*, defaults to `""`) : The begin of sequence token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

sep_token (`str`, *optional*, defaults to `""`): The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

sp_model_kwargs (`dict`, *optional*) : Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:  - `enable_sampling`: Enable subword regularization. - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.  - `nbest_size = {0,1}`: No sampling is performed. - `nbest_size > 1`: samples from the nbest_size results. - `nbest_size >> from transformers import AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
>>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
>>> config.is_decoder = True
>>> model = BertGenerationDecoder.from_pretrained(
...     "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
... )

>>> inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.logits
```

**Parameters:**

config ([BertGenerationDecoder](/docs/transformers/main/en/model_doc/bert-generation#transformers.BertGenerationDecoder)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BertGenerationConfig](/docs/transformers/main/en/model_doc/bert-generation#transformers.BertGenerationConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
