# Speech2Text2

  

  This model is in maintenance mode only, we don't accept any new PRs changing its code.
  If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
  You can do so by running the following command: `pip install -U transformers==4.40.2`.

  

## Overview

The Speech2Text2 model is used together with [Wav2Vec2](wav2vec2) for Speech Translation models proposed in
[Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://huggingface.co/papers/2104.06678) by
Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.

Speech2Text2 is a *decoder-only* transformer model that can be used with any speech *encoder-only*, such as
[Wav2Vec2](wav2vec2) or [HuBERT](hubert) for Speech-to-Text tasks. Please refer to the
[SpeechEncoderDecoder](speech-encoder-decoder) class on how to combine Speech2Text2 with any speech *encoder-only*
model.

This model was contributed by [Patrick von Platen](https://huggingface.co/patrickvonplaten).

The original code can be found [here](https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266).

## Usage tips

- Speech2Text2 achieves state-of-the-art results on the CoVoST Speech Translation dataset. For more information, see
  the [official models](https://huggingface.co/models?other=speech2text2) .
- Speech2Text2 is always used within the [SpeechEncoderDecoder](speech-encoder-decoder) framework.
- Speech2Text2's tokenizer is based on [fastBPE](https://github.com/glample/fastBPE).

## Inference

Speech2Text2's [SpeechEncoderDecoderModel](/docs/transformers/main/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) model accepts raw waveform input values from speech and
makes use of [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) to translate the input speech
autoregressively to the target language.

The [Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) class is responsible for preprocessing the input speech and
[Speech2Text2Tokenizer](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer) decodes the generated target tokens to the target string. The
[Speech2Text2Processor](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor) wraps [Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) and
[Speech2Text2Tokenizer](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer) into a single instance to both extract the input features and decode the
predicted token ids.

- Step-by-step Speech Translation

```python
>>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
>>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

>>> def map_to_array(example):
...     example["speech"] = example["audio"]["array"]
...     return example

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
>>> generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids)
```

- Speech Translation via Pipelines

  The automatic speech recognition pipeline can also be used to translate speech in just a couple lines of code

```python
>>> from datasets import load_dataset
>>> from transformers import pipeline

>>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> asr = pipeline(
...     "automatic-speech-recognition",
...     model="facebook/s2t-wav2vec2-large-en-de",
...     feature_extractor="facebook/s2t-wav2vec2-large-en-de",
... )

>>> translation_de = asr(librispeech_en[0]["file"])
```

See [model hub](https://huggingface.co/models?filter=speech2text2) to look for Speech2Text2 checkpoints.

## Resources

- [Causal language modeling task guide](../tasks/language_modeling)

## Speech2Text2Config[[transformers.Speech2Text2Config]]

#### transformers.Speech2Text2Config[[transformers.Speech2Text2Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/configuration_speech_to_text_2.py#L24)

This is the configuration class to store the configuration of a [Speech2Text2ForCausalLM](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2ForCausalLM). It is used to
instantiate an Speech2Text2 model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Speech2Text2
[facebook/s2t-wav2vec2-large-en-de](https://huggingface.co/facebook/s2t-wav2vec2-large-en-de) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

>>> # Initializing a Speech2Text2 s2t_transformer_s style configuration
>>> configuration = Speech2Text2Config()

>>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
>>> model = Speech2Text2ForCausalLM(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 50265) : Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [Speech2TextModel](/docs/transformers/main/en/model_doc/speech_to_text#transformers.Speech2TextModel)

d_model (`int`, *optional*, defaults to 1024) : Dimensionality of the layers and the pooler layer.

decoder_layers (`int`, *optional*, defaults to 12) : Number of decoder layers.

decoder_attention_heads (`int`, *optional*, defaults to 16) : Number of attention heads for each attention layer in the Transformer decoder.

decoder_ffn_dim (`int`, *optional*, defaults to 4096) : Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.

activation_function (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.

dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, and pooler.

attention_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the attention probabilities.

activation_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for activations inside the fully connected layer.

init_std (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices. https://huggingface.co/papers/1909.11556>`__ for more details.

decoder_layerdrop (`float`, *optional*, defaults to 0.0) : The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models).

max_target_positions (`int`, *optional*, defaults to 1024) : The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).

## Speech2TextTokenizer[[transformers.Speech2Text2Tokenizer]]

#### transformers.Speech2Text2Tokenizer[[transformers.Speech2Text2Tokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/tokenization_speech_to_text_2.py#L55)

Constructs a Speech2Text2Tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.

batch_decodetransformers.Speech2Text2Tokenizer.batch_decodehttps://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3884[{"name": "sequences", "val": ": Union[list[int], list[list[int]], np.ndarray, torch.Tensor]"}, {"name": "skip_special_tokens", "val": ": bool = False"}, {"name": "clean_up_tokenization_spaces", "val": ": Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) --
  List of tokenized input ids. Can be obtained using the `__call__` method.
- **skip_special_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to remove special tokens in the decoding.
- **clean_up_tokenization_spaces** (`bool`, *optional*) --
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
- **kwargs** (additional keyword arguments, *optional*) --
  Will be passed to the underlying model specific decode method.0`list[str]`The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

**Parameters:**

vocab_file (`str`) : File containing the vocabulary.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sentence token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sentence token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths. 

- ****kwargs** : Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

**Returns:**

``list[str]``

The list of decoded sentences.
#### decode[[transformers.Speech2Text2Tokenizer.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L3918)

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

**Parameters:**

token_ids (`Union[int, list[int], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``str``

The decoded sentence.
#### save_vocabulary[[transformers.Speech2Text2Tokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/tokenization_speech_to_text_2.py#L220)

## Speech2Text2Processor[[transformers.Speech2Text2Processor]]

#### transformers.Speech2Text2Processor[[transformers.Speech2Text2Processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/processing_speech_to_text_2.py#L25)

Constructs a Speech2Text2 processor which wraps a Speech2Text2 feature extractor and a Speech2Text2 tokenizer into
a single processor.

[Speech2Text2Processor](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor) offers all the functionalities of [AutoFeatureExtractor](/docs/transformers/main/en/model_doc/auto#transformers.AutoFeatureExtractor) and [Speech2Text2Tokenizer](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer).
See the [__call__()](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

__call__transformers.Speech2Text2Processor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/processing_speech_to_text_2.py#L48[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]

When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
`__call__()` and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to
Speech2Text2Tokenizer's [__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two
methods for more information.

**Parameters:**

feature_extractor (`AutoFeatureExtractor`) : An instance of [AutoFeatureExtractor](/docs/transformers/main/en/model_doc/auto#transformers.AutoFeatureExtractor). The feature extractor is a required input.

tokenizer (`Speech2Text2Tokenizer`) : An instance of [Speech2Text2Tokenizer](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer). The tokenizer is a required input.
#### from_pretrained[[transformers.Speech2Text2Processor.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1360)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### save_pretrained[[transformers.Speech2Text2Processor.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L782)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### batch_decode[[transformers.Speech2Text2Processor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1499)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### decode[[transformers.Speech2Text2Processor.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1508)

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## Speech2Text2ForCausalLM[[transformers.Speech2Text2ForCausalLM]]

#### transformers.Speech2Text2ForCausalLM[[transformers.Speech2Text2ForCausalLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py#L622)

The Speech2Text2 Decoder with a language modeling head. Can be used as the decoder part of [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) and `SpeechEncoderDecoder`.
This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Speech2Text2ForCausalLM.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/deprecated/speech_to_text_2/modeling_speech_to_text_2.py#L648[{"name": "input_ids", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "encoder_hidden_states", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "encoder_attention_mask", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "past_key_values", "val": ": typing.Optional[transformers.cache_utils.Cache] = None"}, {"name": "inputs_embeds", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.LongTensor] = None"}, {"name": "use_cache", "val": ": typing.Optional[bool] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}]- **input_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
  provide it.

  Indices can be obtained using [Speech2Text2Tokenizer](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **encoder_hidden_states**  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) --
  Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
  if the model is configured as a decoder.
- **encoder_attention_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
  in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
- **past_key_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) --
  Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
  shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
  shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
  tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
  cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

  If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
  that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
  all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
- **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
  config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
- **use_cache** (`bool`, *optional*) --
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
  (see `past_key_values`).

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under
  returned tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
  for more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Speech2Text2Config](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config)) and inputs.

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

Example:

```python
>>> from transformers import (
...     SpeechEncoderDecoderModel,
...     Speech2Text2ForCausalLM,
...     Wav2Vec2Model,
...     Speech2Text2Config,
...     Wav2Vec2Config,
...     Wav2Vec2FeatureExtractor,
...     Speech2Text2Tokenizer,
... )
>>> from datasets import load_dataset

>>> feature_extractor = Wav2Vec2FeatureExtractor()
>>> tokenizer = Speech2Text2Tokenizer.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

>>> encoder = Wav2Vec2Model(Wav2Vec2Config())
>>> decoder = Speech2Text2ForCausalLM(Speech2Text2Config())
>>> # init random speech2text model

>>> model = SpeechEncoderDecoderModel(encoder=encoder, decoder=decoder)
>>> model.config.pad_token_id = tokenizer.pad_token_id
>>> model.config.decoder_start_token_id = tokenizer.bos_token_id
>>> # pre-process inputs and labels

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_values = inputs.input_values
>>> decoder_input_ids = tokenizer(ds[0]["text"], return_tensors="pt").input_ids
>>> # compute loss

>>> loss = model(inputs=input_values, labels=decoder_input_ids).loss
>>> # backprop loss

>>> loss.backward()
```

**Parameters:**

config ([Speech2Text2Config](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Speech2Text2Config](/docs/transformers/main/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config)) and inputs.

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
