# Wav2Vec2

## Overview

The Wav2Vec2 model was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://huggingface.co/papers/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

The abstract from the paper is the following:

*We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on
transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks
the speech input in the latent space and solves a contrastive task defined over a quantization of the latent
representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the
clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state
of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and
pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech
recognition with limited amounts of labeled data.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

Note: Meta (FAIR) released a new version of [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - it's pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, e.g. as per [this guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Usage tips

- Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
- Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded
  using [Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

## Using Flash Attention 2

Flash Attention 2 is an faster, optimized version of the model.

### Installation

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:

```bash
pip install -U flash-attn --no-build-isolation
```

### Usage

To load a model using Flash Attention 2, we can pass the argument `attn_implementation="flash_attention_2"` to [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). We'll also load the model in half-precision (e.g. `torch.float16`), since it results in almost no degradation to audio quality but significantly lower memory usage and faster inference:

```python
>>> from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### Expected speedups

Below is an expected speedup diagram comparing the pure inference time between the native implementation in transformers of the `facebook/wav2vec2-large-960h-lv60-self` model and the flash-attention-2 and sdpa (scale-dot-product-attention) versions. . We show the average speedup obtained on the `librispeech_asr` `clean` validation split:

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Wav2Vec2. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A notebook on how to [leverage a pretrained Wav2Vec2 model for emotion classification](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb). 🌎
- [Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).
- [Audio classification task guide](../tasks/audio_classification)

- A blog post on [boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram).
- A blog post on how to [finetune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english).
- A blog post on [finetuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2).
- A notebook on how to [create YouTube captions from any video by transcribing audio with Wav2Vec2](https://colab.research.google.com/github/Muennighoff/ytclipcc/blob/main/wav2vec_youtube_captions.ipynb). 🌎
- [Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by a notebook on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).
- [Automatic speech recognition task guide](../tasks/asr)

🚀 Deploy

- A blog post on how to deploy Wav2Vec2 for [Automatic Speech Recognition with Hugging Face's Transformers & Amazon SageMaker](https://www.philschmid.de/automatic-speech-recognition-sagemaker).

## Wav2Vec2Config[[transformers.Wav2Vec2Config]]

#### transformers.Wav2Vec2Config[[transformers.Wav2Vec2Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/configuration_wav2vec2.py#L27)

This is the configuration class to store the configuration of a [Wav2Vec2Model](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Model). It is used to instantiate an
Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Wav2Vec2
[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) architecture.

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Wav2Vec2Config, Wav2Vec2Model

>>> # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
>>> configuration = Wav2Vec2Config()

>>> # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
>>> model = Wav2Vec2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 32) : Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [Wav2Vec2Model](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Model). Vocabulary size of the model. Defines the different tokens that can be represented by the *inputs_ids* passed to the forward method of [Wav2Vec2Model](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Model).

hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the encoder layers and the pooler layer.

num_hidden_layers (`int`, *optional*, defaults to 12) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 12) : Number of attention heads for each attention layer in the Transformer encoder.

intermediate_size (`int`, *optional*, defaults to 3072) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

activation_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for activations inside the fully connected layer.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

final_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for the final projection layer of [Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC).

layerdrop (`float`, *optional*, defaults to 0.1) : The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-12) : The epsilon used by the layer normalization layers.

feat_extract_norm (`str`, *optional*, defaults to `"group"`) : The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D convolutional layers.

feat_proj_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for output of the feature encoder.

feat_extract_activation (`str, `optional`, defaults to `"gelu"`) : The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

feat_quantizer_dropout (`float`, *optional*, defaults to 0.0) : The dropout probability for quantized feature encoder states.

conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) : A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.

conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) : A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`) : A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The length of *conv_kernel* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_bias (`bool`, *optional*, defaults to `False`) : Whether the 1D convolutional layers have a bias.

num_conv_pos_embeddings (`int`, *optional*, defaults to 128) : Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional embeddings layer.

num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16) : Number of groups of 1D convolutional positional embeddings layer.

do_stable_layer_norm (`bool`, *optional*, defaults to `False`) : Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is False` corresponds to applying layer norm after the attention layer.

apply_spec_augment (`bool`, *optional*, defaults to `True`) : Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://huggingface.co/papers/1904.08779).

mask_time_prob (`float`, *optional*, defaults to 0.05) : Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking procedure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector span to be masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.

mask_time_length (`int`, *optional*, defaults to 10) : Length of vector span along the time axis.

mask_time_min_masks (`int`, *optional*, defaults to 2), : The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step, irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length = 7.5` (Volta).
- **padding_side** (`str`, *optional*) --
  The side on which the model should have padding applied. Should be selected between ['right', 'left'].
  Default value is picked from the class attribute of the same name.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

**Parameters:**

vocab_file (`str`) : File containing the vocabulary.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sentence token.

eos_token (`str`, *optional*, defaults to `""`) : The end of sentence token.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

word_delimiter_token (`str`, *optional*, defaults to `"|"`) : The token used for defining the end of a word.

do_lower_case (`bool`, *optional*, defaults to `False`) : Whether or not to accept lowercase input and lowercase the output when decoding.

target_lang (`str`, *optional*) : A target language the tokenizer should set by default. `target_lang` has to be defined for multi-lingual, nested vocabulary such as [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all). 

- ****kwargs** : Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)
#### save_vocabulary[[transformers.Wav2Vec2CTCTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L675)
#### decode[[transformers.Wav2Vec2CTCTokenizer.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L570)

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

Example:

```python
>>> # Let's see how to retrieve time steps for a model
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> # import model, feature extractor, tokenizer
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

>>> # load first sample of English common_voice
>>> dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
>>> dataset_iter = iter(dataset)
>>> sample = next(dataset_iter)

>>> # forward sample through model to get greedily predicted transcription ids
>>> input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
>>> logits = model(input_values).logits[0]
>>> pred_ids = torch.argmax(logits, axis=-1)

>>> # retrieve word stamps (analogous commands for `output_char_offsets`)
>>> outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
>>> # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
>>> time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

>>> word_offsets = [
...     {
...         "word": d["word"],
...         "start_time": round(d["start_offset"] * time_offset, 2),
...         "end_time": round(d["end_offset"] * time_offset, 2),
...     }
...     for d in outputs.word_offsets
... ]
>>> # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
>>> # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
>>> word_offsets[:3]
[{'word': 'THE', 'start_time': 0.7, 'end_time': 0.78}, {'word': 'TRICK', 'start_time': 0.88, 'end_time': 1.08}, {'word': 'APPEARS', 'start_time': 1.2, 'end_time': 1.64}]
```

**Parameters:**

token_ids (`Union[int, list[int], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces.

output_char_offsets (`bool`, *optional*, defaults to `False`) : Whether or not to output character offsets. Character offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.    Please take a look at the example below to better understand how to make use of `output_char_offsets`.   

output_word_offsets (`bool`, *optional*, defaults to `False`) : Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed words.    Please take a look at the example below to better understand how to make use of `output_word_offsets`.   

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``str` or `Wav2Vec2CTCTokenizerOutput``

The list of decoded
sentences. Will be a `Wav2Vec2CTCTokenizerOutput` when
`output_char_offsets == True` or `output_word_offsets == True`.
#### batch_decode[[transformers.Wav2Vec2CTCTokenizer.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L500)

Convert a list of lists of token ids into a list of strings by calling decode.

**Parameters:**

sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`) : List of tokenized input ids. Can be obtained using the `__call__` method.

skip_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to remove special tokens in the decoding.

clean_up_tokenization_spaces (`bool`, *optional*) : Whether or not to clean up the tokenization spaces.

output_char_offsets (`bool`, *optional*, defaults to `False`) : Whether or not to output character offsets. Character offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.    Please take a look at the Example of [decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.decode) to better understand how to make use of `output_char_offsets`. [batch_decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.batch_decode) works the same way with batched output.   

output_word_offsets (`bool`, *optional*, defaults to `False`) : Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed words.    Please take a look at the Example of [decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.decode) to better understand how to make use of `output_word_offsets`. [batch_decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.batch_decode) works the same way with batched output.   

kwargs (additional keyword arguments, *optional*) : Will be passed to the underlying model specific decode method.

**Returns:**

``list[str]` or `Wav2Vec2CTCTokenizerOutput``

The list of decoded
sentences. Will be a `Wav2Vec2CTCTokenizerOutput` when
`output_char_offsets == True` or `output_word_offsets == True`.
#### set_target_lang[[transformers.Wav2Vec2CTCTokenizer.set_target_lang]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L189)

Set the target language of a nested multi-lingual dictionary

## Wav2Vec2FeatureExtractor[[transformers.Wav2Vec2FeatureExtractor]]

#### transformers.Wav2Vec2FeatureExtractor[[transformers.Wav2Vec2FeatureExtractor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31)

Constructs a Wav2Vec2 feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

__call__transformers.Wav2Vec2FeatureExtractor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L102[{"name": "raw_speech", "val": ": typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]]"}, {"name": "padding", "val": ": typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False"}, {"name": "max_length", "val": ": typing.Optional[int] = None"}, {"name": "truncation", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": typing.Optional[int] = None"}, {"name": "return_attention_mask", "val": ": typing.Optional[bool] = None"}, {"name": "return_tensors", "val": ": typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None"}, {"name": "sampling_rate", "val": ": typing.Optional[int] = None"}, {"name": "**kwargs", "val": ""}]- **raw_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) --
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Select a strategy to pad the returned sequences (according to the model's padding side and padding
  index) among:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **max_length** (`int`, *optional*) --
  Maximum length of the returned list and optionally padding length (see above).
- **truncation** (`bool`) --
  Activates truncation to cut input sequences longer than *max_length* to *max_length*.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature_extractor's default.

  [What are attention masks?](../glossary#attention-mask)

  

  Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
  [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
  `attention_mask`. For such models, `input_values` should simply be padded with 0 and no
  `attention_mask` should be passed.

  For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
  [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should
  be passed for batched inference.

  

- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.
- **sampling_rate** (`int`, *optional*) --
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.
- **padding_value** (`float`, *optional*, defaults to 0.0) --0

Main method to featurize and prepare for the model one or several sequence(s).

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The feature dimension of the extracted features.

sampling_rate (`int`, *optional*, defaults to 16000) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

padding_value (`float`, *optional*, defaults to 0.0) : The value that is used to fill the padding values.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly improve the performance for some models, *e.g.*, [wav2vec2-lv60](https://huggingface.co/models?search=lv60).

return_attention_mask (`bool`, *optional*, defaults to `False`) : Whether or not [__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) should return `attention_mask`.    Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask` should be passed.  For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be passed for batched inference.  

## Wav2Vec2Processor[[transformers.Wav2Vec2Processor]]

#### transformers.Wav2Vec2Processor[[transformers.Wav2Vec2Processor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py#L30)

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
processor.

[Wav2Vec2Processor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) offers all the functionalities of [Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend).
See the docstring of [__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) and [decode()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

__call__transformers.Wav2Vec2Processor.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py#L48[{"name": "audio", "val": ": typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], NoneType] = None"}, {"name": "text", "val": ": typing.Union[str, list[str], NoneType] = None"}, {"name": "**kwargs", "val": ": typing_extensions.Unpack[transformers.models.wav2vec2.processing_wav2vec2.Wav2Vec2ProcessorKwargs]"}]- **audio** (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*) --
  An audio input is passed to [Wav2Vec2FeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__).
- **text** (`str`, `List[str]`, *optional*) --
  A text input is passed to [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__).0This method returns the results of each `call` method. If both are used, the output is a dictionary containing the results of both.

This method forwards all arguments to [Wav2Vec2FeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and/or
[PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) depending on the input modality and returns their outputs. If both modalities are passed, [Wav2Vec2FeatureExtractor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) are called.

**Parameters:**

feature_extractor (`Wav2Vec2FeatureExtractor`) : An instance of [Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). The feature extractor is a required input.

tokenizer ([PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)) : An instance of [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend). The tokenizer is a required input.

**Returns:**

This method returns the results of each `call` method. If both are used, the output is a dictionary containing the results of both.
#### pad[[transformers.Wav2Vec2Processor.pad]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/processing_wav2vec2.py#L94)

This method operates on batches of extracted features and/or tokenized text. It forwards all arguments to
[Wav2Vec2FeatureExtractor.pad()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and/or [PreTrainedTokenizer.pad()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad) depending on the input modality and returns their outputs. If both modalities are passed, [Wav2Vec2FeatureExtractor.pad()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and [PreTrainedTokenizer.pad()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad) are called.

**Parameters:**

input_features : When the first argument is a dictionary containing a batch of tensors, or the `input_features` argument is present, it is passed to [Wav2Vec2FeatureExtractor.pad()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad).

labels : When the `label` argument is present, it is passed to [PreTrainedTokenizer.pad()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad).

**Returns:**

This method returns the results of each `pad` method. If both are used, the output is a dictionary containing the results of both.
#### from_pretrained[[transformers.Wav2Vec2Processor.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1349)

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
#### save_pretrained[[transformers.Wav2Vec2Processor.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L778)

Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
can be reloaded using the [from_pretrained()](/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### batch_decode[[transformers.Wav2Vec2Processor.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1520)

This method forwards all its arguments to PreTrainedTokenizer's [batch_decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.
#### decode[[transformers.Wav2Vec2Processor.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/processing_utils.py#L1529)

This method forwards all its arguments to PreTrainedTokenizer's [decode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## Wav2Vec2ProcessorWithLM[[transformers.Wav2Vec2ProcessorWithLM]]

#### transformers.Wav2Vec2ProcessorWithLM[[transformers.Wav2Vec2ProcessorWithLM]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L69)

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
with language model support into a single processor for language model boosted speech recognition decoding.

__call__transformers.Wav2Vec2ProcessorWithLM.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L217[{"name": "*args", "val": ""}, {"name": "**kwargs", "val": ""}]

When used in normal mode, this method forwards all its arguments to the feature extractor's
`__call__()` and returns its output. If used in the context
`~Wav2Vec2ProcessorWithLM.as_target_processor` this method forwards all its arguments to
Wav2Vec2CTCTokenizer's [__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two
methods for more information.

**Parameters:**

feature_extractor ([Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) or [SeamlessM4TFeatureExtractor](/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor)) : An instance of [Wav2Vec2FeatureExtractor](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) or [SeamlessM4TFeatureExtractor](/docs/transformers/main/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor). The feature extractor is a required input.

tokenizer ([Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer)) : An instance of [Wav2Vec2CTCTokenizer](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer). The tokenizer is a required input.

decoder (`pyctcdecode.BeamSearchDecoderCTC`) : An instance of `pyctcdecode.BeamSearchDecoderCTC`. The decoder is a required input.
#### pad[[transformers.Wav2Vec2ProcessorWithLM.pad]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L252)

When used in normal mode, this method forwards all its arguments to the feature extractor's
`~FeatureExtractionMixin.pad` and returns its output. If used in the context
`~Wav2Vec2ProcessorWithLM.as_target_processor` this method forwards all its arguments to
Wav2Vec2CTCTokenizer's [pad()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad). Please refer to the docstring of the above two methods
for more information.
#### from_pretrained[[transformers.Wav2Vec2ProcessorWithLM.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L115)

Instantiate a [Wav2Vec2ProcessorWithLM](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM) from a pretrained Wav2Vec2 processor.

This class method is simply calling the feature extractor's
[from_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), Wav2Vec2CTCTokenizer's
[from_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), and
`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`.

Please refer to the docstrings of the methods above for more information.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : This can be either:  - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on huggingface.co. - a path to a *directory* containing a feature extractor file saved using the [save_pretrained()](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`. - a path or url to a saved feature extractor JSON *file*, e.g., `./my_model_directory/preprocessor_config.json`.

- ****kwargs** : Additional keyword arguments passed along to both [SequenceFeatureExtractor](/docs/transformers/main/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend)
#### save_pretrained[[transformers.Wav2Vec2ProcessorWithLM.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L111)
#### batch_decode[[transformers.Wav2Vec2ProcessorWithLM.batch_decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L279)

Batch decode output logits to audio transcription with language model support.

This function makes use of Python's multiprocessing. Currently, multiprocessing is available only on Unix
systems (see this [issue](https://github.com/kensho-technologies/pyctcdecode/issues/65)).

If you are decoding multiple batches, consider creating a `Pool` and passing it to `batch_decode`. Otherwise,
`batch_decode` will be very slow since it will create a fresh `Pool` for each call. See usage example below.

Example:
See [Decoding multiple audios](#decoding-multiple-audios).

**Parameters:**

logits (`np.ndarray`) : The logits output vector of the model representing the log probabilities for each token.

pool (`multiprocessing.Pool`, *optional*) : An optional user-managed pool. If not set, one will be automatically created and closed. The pool should be instantiated *after* `Wav2Vec2ProcessorWithLM`. Otherwise, the LM won't be available to the pool's sub-processes.    Currently, only pools created with a 'fork' context can be used. If a 'spawn' pool is passed, it will be ignored and sequential decoding will be used instead.   

num_processes (`int`, *optional*) : If `pool` is not set, number of processes on which the function should be parallelized over. Defaults to the number of available CPUs.

beam_width (`int`, *optional*) : Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.

beam_prune_logp (`int`, *optional*) : Beams that are much worse than best beam will be pruned Defaults to pyctcdecode's DEFAULT_PRUNE_LOGP.

token_min_logp (`int`, *optional*) : Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode's DEFAULT_MIN_TOKEN_LOGP.

hotwords (`list[str]`, *optional*) : List of words with extra importance, can be OOV for LM

hotword_weight (`int`, *optional*) : Weight factor for hotword importance Defaults to pyctcdecode's DEFAULT_HOTWORD_WEIGHT.

alpha (`float`, *optional*) : Weight for language model during shallow fusion

beta (`float`, *optional*) : Weight for length score adjustment of during scoring

unk_score_offset (`float`, *optional*) : Amount of log score offset for unknown tokens

lm_score_boundary (`bool`, *optional*) : Whether to have kenlm respect boundaries when scoring

output_word_offsets (`bool`, *optional*, defaults to `False`) : Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate and model downsampling rate to compute the time-stamps of transcribed words.

n_best (`int`, *optional*, defaults to `1`) : Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list of lists of strings, `logit_score` will be a list of lists of floats, and `lm_score` will be a list of lists of floats, where the length of the outer list will correspond to the batch size and the length of the inner list will correspond to the number of returned hypotheses . The value should be >= 1.    Please take a look at the Example of [decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.decode) to better understand how to make use of `output_word_offsets`. [batch_decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) works the same way with batched output.  

**Returns:**

`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`.
#### decode[[transformers.Wav2Vec2ProcessorWithLM.decode]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L464)

Decode output logits to audio transcription with language model support.

Example:

```python
>>> # Let's see how to retrieve time steps for a model
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> # import model, feature extractor, tokenizer
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> # load first sample of English common_voice
>>> dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
>>> dataset_iter = iter(dataset)
>>> sample = next(dataset_iter)

>>> # forward sample through model to get greedily predicted transcription ids
>>> input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values
>>> with torch.no_grad():
...     logits = model(input_values).logits[0].cpu().numpy()

>>> # retrieve word stamps (analogous commands for `output_char_offsets`)
>>> outputs = processor.decode(logits, output_word_offsets=True)
>>> # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
>>> time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

>>> word_offsets = [
...     {
...         "word": d["word"],
...         "start_time": round(d["start_offset"] * time_offset, 2),
...         "end_time": round(d["end_offset"] * time_offset, 2),
...     }
...     for d in outputs.word_offsets
... ]
>>> # compare word offsets with audio `en_train_0/common_voice_en_19121553.mp3` online on the dataset viewer:
>>> # https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/en
>>> word_offsets[:4]
[{'word': 'THE', 'start_time': 0.68, 'end_time': 0.78}, {'word': 'TRACK', 'start_time': 0.88, 'end_time': 1.1}, {'word': 'APPEARS', 'start_time': 1.18, 'end_time': 1.66}, {'word': 'ON', 'start_time': 1.86, 'end_time': 1.92}]
```

**Parameters:**

logits (`np.ndarray`) : The logits output vector of the model representing the log probabilities for each token.

beam_width (`int`, *optional*) : Maximum number of beams at each step in decoding. Defaults to pyctcdecode's DEFAULT_BEAM_WIDTH.

beam_prune_logp (`int`, *optional*) : A threshold to prune beams with log-probs less than best_beam_logp + beam_prune_logp. The value should be = 1.    Please take a look at the example below to better understand how to make use of `output_word_offsets`.  

**Returns:**

`~models.wav2vec2.Wav2Vec2DecoderWithLMOutput`.

### Decoding multiple audios

If you are planning to decode multiple batches of audios, you should consider using [batch_decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) and passing an instantiated `multiprocessing.Pool`.
Otherwise, [batch_decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) performance will be slower than calling [decode()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.decode) for each audio individually, as it internally instantiates a new `Pool` for every call. See the example below:

```python
>>> # Let's see how to use a user-managed pool for batch decoding multiple audios
>>> from multiprocessing import get_context
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
from accelerate import Accelerator
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> device = Accelerator().device
>>> # import model, feature extractor, tokenizer
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm").to(device)
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> # load example dataset
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))

>>> def map_to_array(example):
...     example["speech"] = example["audio"]["array"]
...     return example

>>> # prepare speech data for batch inference
>>> dataset = dataset.map(map_to_array, remove_columns=["audio"])

>>> def map_to_pred(batch, pool):
...     device = Accelerator().device
...     inputs = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")
...     inputs = {k: v.to(device) for k, v in inputs.items()}

...     with torch.no_grad():
...         logits = model(**inputs).logits

...     transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
...     batch["transcription"] = transcription
...     return batch

>>> # note: pool should be instantiated *after* `Wav2Vec2ProcessorWithLM`.
>>> #       otherwise, the LM won't be available to the pool's sub-processes
>>> # select number of processes and batch_size based on number of CPU cores available and on dataset size
>>> with get_context("fork").Pool(processes=2) as pool:
...     result = dataset.map(
...         map_to_pred, batched=True, batch_size=2, fn_kwargs={"pool": pool}, remove_columns=["speech"]
...     )

>>> result["transcription"][:2]
['MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL', "NOR IS MISTER COULTER'S MANNER LESS INTERESTING THAN HIS MATTER"]
```

## Wav2Vec2 specific outputs[[transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput]]

#### transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput[[transformers.models.wav2vec2_with_lm.processing_wav2vec2_with_lm.Wav2Vec2DecoderWithLMOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L47)

Output type of `Wav2Vec2DecoderWithLM`, with transcription.

**Parameters:**

text (list of `str` or `str`) : Decoded logits in text from. Usually the speech transcription.

logit_score (list of `float` or `float`) : Total logit score of the beams associated with produced text.

lm_score (list of `float`) : Fused lm_score of the beams associated with produced text.

word_offsets (list of `list[dict[str, Union[int, str]]]` or `list[dict[str, Union[int, str]]]`) : Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets can be used to compute time stamps for each word.

#### transformers.modeling_outputs.Wav2Vec2BaseModelOutput[[transformers.modeling_outputs.Wav2Vec2BaseModelOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L1347)

Base class for models that have been trained with the Wav2Vec2 loss objective.

**Parameters:**

last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) : Sequence of hidden-states at the output of the last layer of the model.

extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) : Sequence of extracted feature vectors of the last convolutional layer of the model.

hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the initial embedding outputs.

attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

#### transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput[[transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L75)

Output type of [Wav2Vec2ForPreTraining](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining), with potential hidden states and attentions.

**Parameters:**

loss (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) : Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).

projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) : Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked projected quantized states.

projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) : Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive target vectors for contrastive loss.

codevector_perplexity (`torch.FloatTensor` of shape `(1,)`) : The perplexity of the codevector distribution, used to measure the diversity of the codebook.

hidden_states (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) : Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

attentions (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) : Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

contrastive_loss (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) : The contrastive loss (L_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).

diversity_loss (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) : The diversity loss (L_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).

## Wav2Vec2Model[[transformers.Wav2Vec2Model]]

#### transformers.Wav2Vec2Model[[transformers.Wav2Vec2Model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1260)

The bare Wav2Vec2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2Model.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1334[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mask_time_indices", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mask_time_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
  masked extracted features in *config.proj_codevector_dim* space.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **extract_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) -- Sequence of extracted feature vectors of the last convolutional layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Wav2Vec2Model](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

**Parameters:**

config ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **extract_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) -- Sequence of extracted feature vectors of the last convolutional layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Wav2Vec2ForCTC[[transformers.Wav2Vec2ForCTC]]

#### transformers.Wav2Vec2ForCTC[[transformers.Wav2Vec2ForCTC]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1660)

Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2ForCTC.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1726[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*) --
  Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
  the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
  All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
  config.vocab_size - 1]`.0[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoProcessor, Wav2Vec2ForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

**Parameters:**

config ([Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

target_lang (`str`, *optional*) : Language id of adapter weights. Adapter weights are stored in the format adapter..safetensors or adapter..bin. Only relevant when using an instance of [Wav2Vec2ForCTC](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) with adapters. Uses 'eng' by default.

**Returns:**

`[transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.CausalLMOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
#### load_adapter[[transformers.Wav2Vec2ForCTC.load_adapter]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1083)

Load a language adapter model from a pre-trained adapter model.

Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
use this method in a firewalled environment.

Examples:

```python
>>> from transformers import Wav2Vec2ForCTC, AutoProcessor

>>> ckpt = "facebook/mms-1b-all"
>>> processor = AutoProcessor.from_pretrained(ckpt)
>>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
>>> # set specific language
>>> processor.tokenizer.set_target_lang("spa")
>>> model.load_adapter("spa")
```

**Parameters:**

target_lang (`str`) : Has to be a language id of an existing adapter weight. Adapter weights are stored in the format adapter..safetensors or adapter..bin

force_load (`bool`, defaults to `True`) : Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.

cache_dir (`Union[str, os.PathLike]`, *optional*) : Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.

proxies (`dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

local_files_only(`bool`, *optional*, defaults to `False`) : Whether or not to only look at local files (i.e., do not try to download the model).

token (`str` or `bool`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.    To test a pull request you made on the Hub, you can pass `revision="refs/pr/"`.   

mirror (`str`, *optional*) : Mirror source to accelerate downloads in China. If you are from China and have an accessibility problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety. Please refer to the mirror site for more information.

## Wav2Vec2ForSequenceClassification[[transformers.Wav2Vec2ForSequenceClassification]]

#### transformers.Wav2Vec2ForSequenceClassification[[transformers.Wav2Vec2ForSequenceClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1805)

Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2ForSequenceClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1838[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Wav2Vec2ForSequenceClassification](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, Wav2Vec2ForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, Wav2Vec2ForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = Wav2Vec2ForSequenceClassification.from_pretrained(
...     "facebook/wav2vec2-base-960h", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

**Parameters:**

config ([Wav2Vec2ForSequenceClassification](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.SequenceClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Wav2Vec2ForAudioFrameClassification[[transformers.Wav2Vec2ForAudioFrameClassification]]

#### transformers.Wav2Vec2ForAudioFrameClassification[[transformers.Wav2Vec2ForAudioFrameClassification]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1910)

The Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2ForAudioFrameClassification.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1942[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Wav2Vec2ForAudioFrameClassification](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForAudioFrameClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForAudioFrameClassification.from_pretrained("facebook/wav2vec2-base-960h")

>>> # audio file is decoded on the fly
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> probabilities = torch.sigmoid(logits[0])
>>> # labels is a one-hot array of shape (num_frames, num_speakers)
>>> labels = (probabilities > 0.5).long()
>>> labels[0].tolist()
...
```

**Parameters:**

config ([Wav2Vec2ForAudioFrameClassification](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.TokenClassifierOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided)  -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Wav2Vec2ForXVector[[transformers.Wav2Vec2ForXVector]]

#### transformers.Wav2Vec2ForXVector[[transformers.Wav2Vec2ForXVector]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2064)

Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2ForXVector.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2114[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) --
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
  config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).0[transformers.modeling_outputs.XVectorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`A [transformers.modeling_outputs.XVectorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) -- Classification hidden states before AMSoftmax.
- **embeddings** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) -- Utterance embeddings used for vector similarity-based retrieval.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [Wav2Vec2ForXVector](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForXVector.from_pretrained("facebook/wav2vec2-base-960h")

>>> # audio file is decoded on the fly
>>> inputs = feature_extractor(
...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
... )
>>> with torch.no_grad():
...     embeddings = model(**inputs).embeddings

>>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

>>> # the resulting embeddings can be used for cosine similarity-based retrieval
>>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
>>> similarity = cosine_sim(embeddings[0], embeddings[1])
>>> threshold = 0.7  # the optimal threshold is dataset-dependent
>>> if similarity >> round(similarity.item(), 2)
...
```

**Parameters:**

config ([Wav2Vec2ForXVector](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.modeling_outputs.XVectorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)``

A [transformers.modeling_outputs.XVectorOutput](/docs/transformers/main/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) -- Classification hidden states before AMSoftmax.
- **embeddings** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) -- Utterance embeddings used for vector similarity-based retrieval.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Wav2Vec2ForPreTraining[[transformers.Wav2Vec2ForPreTraining]]

#### transformers.Wav2Vec2ForPreTraining[[transformers.Wav2Vec2ForPreTraining]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1399)

Wav2Vec2 Model with a quantizer and `VQ` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.Wav2Vec2ForPreTraining.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1447[{"name": "input_values", "val": ": typing.Optional[torch.Tensor]"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "mask_time_indices", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "sampled_negative_indices", "val": ": typing.Optional[torch.BoolTensor] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "**kwargs", "val": ""}]- **input_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.__call__()](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **mask_time_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
  masked extracted features in *config.proj_codevector_dim* space.
- **sampled_negative_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*) --
  Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
  Required input for pre-training.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.0[transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)`A [transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
- **projected_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
  projected quantized states.
- **projected_quantized_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
  target vectors for contrastive loss.
- **codevector_perplexity** (`torch.FloatTensor` of shape `(1,)`) -- The perplexity of the codevector distribution, used to measure the diversity of the codebook.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **contrastive_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- The contrastive loss (L_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
- **diversity_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- The diversity loss (L_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
The [Wav2Vec2ForPreTraining](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> import torch
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
>>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
>>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

>>> # compute masked indices
>>> batch_size, raw_sequence_length = input_values.shape
>>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
>>> mask_time_indices = _compute_mask_indices(
...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
... )
>>> sampled_negative_indices = _sample_negative_indices(
...     features_shape=(batch_size, sequence_length),
...     num_negatives=model.config.num_negatives,
...     mask_time_indices=mask_time_indices,
... )
>>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
>>> sampled_negative_indices = torch.tensor(
...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
... )

>>> with torch.no_grad():
...     outputs = model(input_values, mask_time_indices=mask_time_indices)

>>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
>>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

>>> # show that cosine similarity is much higher than random
>>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
tensor(True)

>>> # for contrastive loss training model should be put into train mode
>>> model = model.train()
>>> loss = model(
...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
... ).loss
```

**Parameters:**

config ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

`[transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)``

A [transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/main/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/main/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

- **loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
- **projected_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
  projected quantized states.
- **projected_quantized_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) -- Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
  target vectors for contrastive loss.
- **codevector_perplexity** (`torch.FloatTensor` of shape `(1,)`) -- The perplexity of the codevector distribution, used to measure the diversity of the codebook.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **contrastive_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- The contrastive loss (L_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
- **diversity_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) -- The diversity loss (L_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
