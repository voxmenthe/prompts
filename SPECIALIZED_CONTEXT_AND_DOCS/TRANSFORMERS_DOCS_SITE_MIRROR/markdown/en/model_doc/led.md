# LED

[Longformer-Encoder-Decoder (LED)](https://huggingface.co/papers/2004.05150) is an encoder-decoder  transformer model for sequence-to-sequence tasks like summarization. It extends [Longformer](.longformer), an encoder-only model designed to handle long inputs, by adding a decoder layer. The decoder uses full self-attention on the encoded tokens and previously decoded locations. Because of Longformer's linear self-attention mechanism, LED is more efficient than standard encoder-decoder models when processing long sequences.

You can find all the original [LED] checkpoints under the [Ai2](https://huggingface.co/allenai/models?search=led) organization.

> [!TIP]
> This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).
>
> Click on the LED models in the right sidebar for more examples of how to apply LED to different language tasks.

The example below demonstrates how to summarize text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
    task="summarization",
    model="allenai/led-base-16384",
    dtype=torch.float16,
    device=0
)
pipeline("""Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle.""")
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/led-base-16384"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "allenai/led-base-16384",
    dtype=torch.float16,
    device_map="auto"
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

# Place global attention on the first token
global_attention_mask = torch.zeros_like(input_ids.input_ids).to(model.device)
global_attention_mask[:, 0] = 1

output = model.generate(**input_ids, global_attention_mask=global_attention_mask, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

```bash
!echo -e "Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet. Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts." | transformers run --task summarization --model allenai/led-base-16384 --device 0
```

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "allenai/led-large-16384",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    "allenai/led-large-16384"
)

input_text = """Plants are among the most remarkable and essential life forms on Earth, possessing a unique ability to produce their own food through a process known as photosynthesis. This complex biochemical process is fundamental not only to plant life but to virtually all life on the planet.
Through photosynthesis, plants capture energy from sunlight using a green pigment called chlorophyll, which is located in specialized cell structures called chloroplasts. In the presence of light, plants absorb carbon dioxide from the atmosphere through small pores in their leaves called stomata, and take in water from the soil through their root systems.
These ingredients are then transformed into glucose, a type of sugar that serves as a source of chemical energy, and oxygen, which is released as a byproduct into the atmosphere. The glucose produced during photosynthesis is not just used immediately; plants also store it as starch or convert it into other organic compounds like cellulose, which is essential for building their cellular structure.
This energy reserve allows them to grow, develop leaves, produce flowers, bear fruit, and carry out various physiological processes throughout their lifecycle."""
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

# Place global attention on the first token
global_attention_mask = torch.zeros_like(input_ids.input_ids).to(model.device)
global_attention_mask[:, 0] = 1

output = model.generate(**input_ids, global_attention_mask=global_attention_mask, cache_implementation="static")
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Notes

- [LEDForConditionalGeneration](/docs/transformers/main/en/model_doc/led#transformers.LEDForConditionalGeneration) is an extension of [BartForConditionalGeneration](/docs/transformers/main/en/model_doc/bart#transformers.BartForConditionalGeneration) exchanging the traditional self-attention layer with Longformer's chunked self-attention layer. [LEDTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer) is an alias of [BartTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer).
- LED pads the `input_ids` to be a multiple of `config.attention_window` if required. A small speedup is gained when [LEDTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer) is used with the `pad_to_multiple_of` argument.
- LED works best on long-range sequence-to-sequence tasks where the `input_ids` are significantly longer than 1024 tokens.
- LED uses global attention by means of the `global_attention_mask` (see [LongformerModel](/docs/transformers/main/en/model_doc/longformer#transformers.LongformerModel)). For summarization, it is advised to put global attention only on the first `` token. For question answering, it is advised to put global attention on all tokens of the question.
- To fine-tune LED on all 16384 parameters, gradient checkpointing can be enabled in case training leads to out-of-memory (OOM) errors. Enable gradient checkpointing by adding `model.gradient_checkpointing_enable()` and setting `use_cache=False` to disable the caching mechanism to save memory.
- Inputs should be padded on the right because LED uses absolute position embeddings.

## Resources

- Read the [LED on Arxiv notebook](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing) to see how LED can achieve state-of-the-art performance on Arxiv article summarization.
- Read the [Fine-tune LED notebook](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing) to learn how to fine-tune LED on PubMed articles.

## LEDConfig[[transformers.LEDConfig]]

#### transformers.LEDConfig[[transformers.LEDConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/configuration_led.py#L26)

This is the configuration class to store the configuration of a [LEDModel](/docs/transformers/main/en/model_doc/led#transformers.LEDModel). It is used to instantiate an LED
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the LED
[allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import LEDModel, LEDConfig

>>> # Initializing a LED allenai/led-base-16384 style configuration
>>> configuration = LEDConfig()

>>> # Initializing a model from the allenai/led-base-16384 style configuration
>>> model = LEDModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50265) : Vocabulary size of the LED model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LEDModel](/docs/transformers/main/en/model_doc/led#transformers.LEDModel).

d_model (`int`, *optional*, defaults to 1024) : Dimensionality of the layers and the pooler layer.

encoder_layers (`int`, *optional*, defaults to 12) : Number of encoder layers.

decoder_layers (`int`, *optional*, defaults to 12) : Number of decoder layers.

encoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer encoder.

decoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer decoder.

decoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.

encoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.

activation_function (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

activation_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for activations inside the fully connected layer.

classifier_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for classifier.

max_encoder_position_embeddings (`int`, *optional*, defaults to 16384) : The maximum sequence length that the encoder might ever be used with.

max_decoder_position_embeddings (`int`, *optional*, defaults to 16384) : The maximum sequence length that the decoder might ever be used with.

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

encoder_layerdrop (`float`, *optional*, defaults to 0.0) : The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

decoder_layerdrop (`float`, *optional*, defaults to 0.0) : The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models)

## LEDTokenizer[[transformers.RobertaTokenizer]]

#### transformers.RobertaTokenizer[[transformers.RobertaTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L31)

Construct a RoBERTa tokenizer (backed by HuggingFace's tokenizers library). Based on Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```python
>>> from transformers import RobertaTokenizer

>>> tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

get_special_tokens_masktransformers.RobertaTokenizer.get_special_tokens_maskhttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1335[{"name": "token_ids_0", "val": ": list[int]"}, {"name": "token_ids_1", "val": ": Optional[list[int]] = None"}, {"name": "already_has_special_tokens", "val": ": bool = False"}]- **token_ids_0** -- List of IDs for the (possibly already formatted) sequence.
- **token_ids_1** -- Unused when `already_has_special_tokens=True`. Must be None in that case.
- **already_has_special_tokens** -- Whether the sequence is already formatted with special tokens.0A list of integers in the range [0, 1]1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added.

For fast tokenizers, data collators call this with `already_has_special_tokens=True` to build a mask over an
already-formatted sequence. In that case, we compute the mask by checking membership in `all_special_ids`.

**Parameters:**

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

merges (`str` or `list`, *optional*) : Custom merges list. If not provided, merges are loaded from merges_file.

errors (`str`, *optional*, defaults to `"replace"`) : Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `False`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (RoBERTa tokenizer detect beginning of words by the preceding space).

trim_offsets (`bool`, *optional*, defaults to `True`) : Whether the post processing step should trim offsets to avoid including whitespaces.

**Returns:**

`A list of integers in the range [0, 1]`

1 for a special token, 0 for a sequence token.
#### save_vocabulary[[transformers.RobertaTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_tokenizers.py#L384)

## LEDTokenizerFast[[transformers.RobertaTokenizer]]

#### transformers.RobertaTokenizer[[transformers.RobertaTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L31)

Construct a RoBERTa tokenizer (backed by HuggingFace's tokenizers library). Based on Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```python
>>> from transformers import RobertaTokenizer

>>> tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [TokenizersBackend](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

**Parameters:**

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

merges (`str` or `list`, *optional*) : Custom merges list. If not provided, merges are loaded from merges_file.

errors (`str`, *optional*, defaults to `"replace"`) : Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

add_prefix_space (`bool`, *optional*, defaults to `False`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (RoBERTa tokenizer detect beginning of words by the preceding space).

trim_offsets (`bool`, *optional*, defaults to `True`) : Whether the post processing step should trim offsets to avoid including whitespaces.

## LED specific outputs[[transformers.models.led.modeling_led.LEDEncoderBaseModelOutput]]

#### transformers.models.led.modeling_led.LEDEncoderBaseModelOutput[[transformers.models.led.modeling_led.LEDEncoderBaseModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1088)

Base class for LEDEncoder's outputs, with potential hidden states, local and global attentions.

**Parameters:**

last_hidden_state (`.last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x + attention_window + 1)`, where `x` is the number of tokens with global attention mask.  Local attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token in the sequence to every token with global attention (first `x` values) and to every token in the attention window (remaining `attention_window + 1` values). Note that the first `x` values refer to tokens with fixed positions in the text, but the remaining `attention_window + 1` values refer to tokens with relative positions: the attention weight of a token to itself is located at index `x + attention_window / 2` and the `attention_window / 2` preceding (succeeding) values are the attention weights to the `attention_window / 2` preceding (succeeding) tokens. If the attention window contains a token with global attention, the attention weight at the corresponding index is set to 0; the value should be accessed from the first `x` attention weights. If a token has global attention, the attention weights to all other tokens in `attentions` is set to 0, the values should be accessed from `global_attentions`.

global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.  Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.

#### transformers.models.led.modeling_led.LEDSeq2SeqModelOutput[[transformers.models.led.modeling_led.LEDSeq2SeqModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1127)

Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
decoding.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the decoder of the model.  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.  Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.

#### transformers.models.led.modeling_led.LEDSeq2SeqLMOutput[[transformers.models.led.modeling_led.LEDSeq2SeqLMOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1165)

Base class for sequence-to-sequence language models outputs.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Language modeling loss.

logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) : Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.  Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.

#### transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput[[transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1203)

Base class for outputs of sequence-to-sequence sentence classification models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) : Classification (or regression if config.num_labels==1) loss.

logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) : Classification (or regression if config.num_labels==1) scores (before SoftMax).

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.  Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.

#### transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput[[transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1241)

Base class for outputs of sequence-to-sequence question answering models.

**Parameters:**

loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) : Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.

start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) : Span-start scores (before SoftMax).

end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) : Span-end scores (before SoftMax).

past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) : It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.

decoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.

decoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

cross_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) : Sequence of hidden-states at the output of the last layer of the encoder of the model.

encoder_hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.

encoder_attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.

encoder_global_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`, where `x` is the number of tokens with global attention mask.  Global attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads. Those are the attention weights from every token with global attention to every token in the sequence.

## LEDModel[[transformers.LEDModel]]

#### transformers.LEDModel[[transformers.LEDModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1750)

The bare Led Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LEDModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1776[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"}, {"name": "global_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `LedTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_led._prepare_decoder_inputs` and modify
  to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more information on the
  default strategy.
- **encoder_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **global_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to decide the attention given on each token, local attention or global attention for the encoder.
  Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is
  important for task-specific finetuning because it makes the model more flexible at representing the task.
  For example, for classification, the  token should be given global attention. For QA, all question
  tokens should also have global attention. Please refer to the [Longformer
  paper](https://huggingface.co/papers/2004.05150) for more details. Mask values selected in `[0, 1]`:

  - 0 for local attention (a sliding window attention),
  - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.models.led.modeling_led.LEDSeq2SeqModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.led.modeling_led.LEDSeq2SeqModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.
The [LEDModel](/docs/transformers/main/en/model_doc/led#transformers.LEDModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.led.modeling_led.LEDSeq2SeqModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.led.modeling_led.LEDSeq2SeqModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

## LEDForConditionalGeneration[[transformers.LEDForConditionalGeneration]]

#### transformers.LEDForConditionalGeneration[[transformers.LEDForConditionalGeneration]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1894)

The LED Model with a language modeling head. Can be used for summarization.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LEDForConditionalGeneration.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L1926[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"}, {"name": "global_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "cache_position", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `LedTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_led._prepare_decoder_inputs` and modify
  to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more information on the
  default strategy.
- **encoder_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **global_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to decide the attention given on each token, local attention or global attention for the encoder.
  Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is
  important for task-specific finetuning because it makes the model more flexible at representing the task.
  For example, for classification, the  token should be given global attention. For QA, all question
  tokens should also have global attention. Please refer to the [Longformer
  paper](https://huggingface.co/papers/2004.05150) for more details. Mask values selected in `[0, 1]`:

  - 0 for local attention (a sliding window attention),
  - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
- **past_key_values** (`~cache_utils.Cache`, *optional*) --
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/main/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **cache_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) --
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.0[transformers.models.led.modeling_led.LEDSeq2SeqLMOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`A [transformers.models.led.modeling_led.LEDSeq2SeqLMOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.
The [LEDForConditionalGeneration](/docs/transformers/main/en/model_doc/led#transformers.LEDForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example Summarization:

```python
>>> import torch
>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

>>> ARTICLE_TO_SUMMARIZE = '''Transformers (Vaswani et al., 2017) have achieved state-of-the-art
...     results in a wide range of natural language tasks including generative language modeling
...     (Dai et al., 2019; Radford et al., 2019) and discriminative ... language understanding (Devlin et al., 2019).
...     This success is partly due to the self-attention component which enables the network to capture contextual
...     information from the entire sequence. While powerful, the memory and computational requirements of
...     self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to
...     process long sequences. To address this limitation, we present Longformer, a modified Transformer
...     architecture with a self-attention operation that scales linearly with the sequence length, making it
...     versatile for processing long documents (Fig 1). This is an advantage for natural language tasks such as
...     long document classification, question answering (QA), and coreference resolution, where existing approaches
...     partition or shorten the long context into smaller sequences that fall within the typical 512 token limit
...     of BERT-style pretrained models. Such partitioning could potentially result in loss of important
...     cross-partition information, and to mitigate this problem, existing methods often rely on complex
...     architectures to address such interactions. On the other hand, our proposed Longformer is able to build
...     contextual representations of the entire context using multiple layers of attention, reducing the need for
...     task-specific architectures.'''
>>> inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

>>> # Global attention on the first token (cf. Beltagy et al. 2020)
>>> global_attention_mask = torch.zeros_like(inputs)
>>> global_attention_mask[:, 0] = 1

>>> # Generate Summary
>>> summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)
>>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
```

Example Conditional generation :

```python
>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> TXT = "My friends are  but they eat too many carbs."

>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

>>> prediction = model.generate(input_ids)[0]
>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
```

**Parameters:**

config ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.led.modeling_led.LEDSeq2SeqLMOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.led.modeling_led.LEDSeq2SeqLMOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

## LEDForSequenceClassification[[transformers.LEDForSequenceClassification]]

#### transformers.LEDForSequenceClassification[[transformers.LEDForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L2091)

LED model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LEDForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L2111[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"}, {"name": "global_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `LedTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_led._prepare_decoder_inputs` and modify
  to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more information on the
  default strategy.
- **encoder_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **global_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to decide the attention given on each token, local attention or global attention for the encoder.
  Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is
  important for task-specific finetuning because it makes the model more flexible at representing the task.
  For example, for classification, the  token should be given global attention. For QA, all question
  tokens should also have global attention. Please refer to the [Longformer
  paper](https://huggingface.co/papers/2004.05150) for more details. Mask values selected in `[0, 1]`:

  - 0 for local attention (a sliding window attention),
  - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.
The [LEDForSequenceClassification](/docs/transformers/main/en/model_doc/led#transformers.LEDForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, LEDForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, LEDForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = LEDForSequenceClassification.from_pretrained(
...     "allenai/led-base-16384", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

**Parameters:**

config ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqSequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.

## LEDForQuestionAnswering[[transformers.LEDForQuestionAnswering]]

#### transformers.LEDForQuestionAnswering[[transformers.LEDForQuestionAnswering]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L2236)

The Led transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.LEDForQuestionAnswering.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/led/modeling_led.py#L2249[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "decoder_input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "decoder_attention_mask", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "encoder_outputs", "val": ": typing.Optional[tuple[tuple[torch.FloatTensor]]] = None"}, {"name": "global_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "start_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "end_positions", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "decoder_inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **decoder_input_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `LedTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)

  LED uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
- **decoder_attention_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) --
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_led._prepare_decoder_inputs` and modify
  to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more information on the
  default strategy.
- **encoder_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) --
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
- **global_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to decide the attention given on each token, local attention or global attention for the encoder.
  Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is
  important for task-specific finetuning because it makes the model more flexible at representing the task.
  For example, for classification, the  token should be given global attention. For QA, all question
  tokens should also have global attention. Please refer to the [Longformer
  paper](https://huggingface.co/papers/2004.05150) for more details. Mask values selected in `[0, 1]`:

  - 0 for local attention (a sliding window attention),
  - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).
- **start_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the start of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **end_positions** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for position (index) of the end of the labelled span for computing the token classification loss.
  Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
  are not taken into account for computing the loss.
- **inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model's internal embedding lookup matrix.
- **decoder_inputs_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) --
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`A [transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) -- Span-end scores (before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.
The [LEDForQuestionAnswering](/docs/transformers/main/en/model_doc/led#transformers.LEDForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoTokenizer, LEDForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForQuestionAnswering.from_pretrained("allenai/led-base-16384")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
...

>>> # target is "nice puppet"
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
...
```

**Parameters:**

config ([LEDForQuestionAnswering](/docs/transformers/main/en/model_doc/led#transformers.LEDForQuestionAnswering)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/main/en/model_doc/led#transformers.models.led.modeling_led.LEDSeq2SeqQuestionAnsweringModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([LEDConfig](/docs/transformers/main/en/model_doc/led#transformers.LEDConfig)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
- **start_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) -- Span-start scores (before SoftMax).
- **end_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to `None`) -- Span-end scores (before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/main/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
  used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **encoder_global_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, x)`,
  where `x` is the number of tokens with global attention mask.

  Global attentions weights after the attention softmax, used to compute the weighted average in the
  self-attention heads. Those are the attention weights from every token with global attention to every token
  in the sequence.
