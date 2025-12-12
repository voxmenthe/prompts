*This model was released on 2022-11-12 and added to Hugging Face Transformers on 2023-02-16.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# CLAP

[CLAP (Contrastive Language-Audio Pretraining)](https://huggingface.co/papers/2211.06687) is a multimodal model that combines audio data with natural language descriptions through contrastive learning.

It incorporates feature fusion and keyword-to-caption augmentation to process variable-length audio inputs and to improve performance. CLAP doesn’t require task-specific training data and can learn meaningful audio representations through natural language.

You can find all the original CLAP checkpoints under the [CLAP](https://huggingface.co/collections/laion/clap-contrastive-language-audio-pretraining-65415c0b18373b607262a490) collection.

This model was contributed by [ybelkada](https://huggingface.co/ybelkada) and [ArthurZ](https://huggingface.co/ArthurZ).

Click on the CLAP models in the right sidebar for more examples of how to apply CLAP to different audio retrieval and classification tasks.

The example below demonstrates how to extract text embeddings with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
import torch
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained("laion/clap-htsat-unfused", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

texts = ["the sound of a cat", "the sound of a dog", "music playing"]

inputs = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

print(f"Text embeddings shape: {text_features.shape}")
print(f"Text embeddings: {text_features}")
```

## ClapConfig

### class transformers.ClapConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/configuration_clap.py#L292)

( text\_config = None audio\_config = None logit\_scale\_init\_value = 14.285714285714285 projection\_dim = 512 projection\_hidden\_act = 'relu' initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [ClapTextConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextConfig).
* **audio\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [ClapAudioConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioConfig).
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 14.29) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original CLAP implementation.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of text and audio projection layers.
* **projection\_hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  Activation function for the projection layers.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  Factor to scale the initialization of the model weights.
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig) is the configuration class to store the configuration of a [ClapModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapModel). It is used to instantiate
a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
configuration with the defaults will yield a similar configuration to that of the CLAP
[laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ClapConfig, ClapModel

>>> # Initializing a ClapConfig with laion-ai/base style configuration
>>> configuration = ClapConfig()

>>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
>>> model = ClapModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
>>> from transformers import ClapTextConfig, ClapAudioConfig

>>> # Initializing a ClapText and ClapAudioConfig configuration
>>> config_text = ClapTextConfig()
>>> config_audio = ClapAudioConfig()

>>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
```

#### from\_text\_audio\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/configuration_utils.py#L1272)

( text\_config audio\_config \*\*kwargs  ) → `PreTrainedConfig`

Returns

`PreTrainedConfig`

An instance of a configuration object

Instantiate a model config (or a derived class) from text model configuration and audio model
configuration.

## ClapTextConfig

### class transformers.ClapTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/configuration_clap.py#L24)

( vocab\_size = 50265 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 514 type\_vocab\_size = 1 initializer\_factor = 1.0 layer\_norm\_eps = 1e-12 projection\_dim = 512 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 position\_embedding\_type = 'absolute' use\_cache = True projection\_hidden\_act = 'relu' \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `Callable`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
  `"relu"`, `"silu"` and `"relu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel).
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **position\_embedding\_type** (`str`, *optional*, defaults to `"absolute"`) —
  Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
  positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
  [Self-Attention with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
  For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
  with Better Relative Position Embeddings (Huang et al.)](https://huggingface.co/papers/2009.13658).
* **is\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **projection\_hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Dimension of the projection head of the `ClapTextModelWithProjection`.

This is the configuration class to store the configuration of a [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel). It is used to instantiate a CLAP
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the CLAP
[calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:


```
>>> from transformers import ClapTextConfig, ClapTextModel

>>> # Initializing a CLAP text configuration
>>> configuration = ClapTextConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = ClapTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ClapAudioConfig

### class transformers.ClapAudioConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/configuration_clap.py#L139)

( window\_size = 8 num\_mel\_bins = 64 spec\_size = 256 hidden\_act = 'gelu' patch\_size = 4 patch\_stride = [4, 4] num\_classes = 527 hidden\_size = 768 projection\_dim = 512 depths = [2, 2, 6, 2] num\_attention\_heads = [4, 8, 16, 32] enable\_fusion = False hidden\_dropout\_prob = 0.1 fusion\_type = None patch\_embed\_input\_channels = 1 flatten\_patch\_embeds = True patch\_embeds\_hidden\_size = 96 enable\_patch\_layer\_norm = True drop\_path\_rate = 0.0 attention\_probs\_dropout\_prob = 0.0 qkv\_bias = True mlp\_ratio = 4.0 aff\_block\_r = 4 num\_hidden\_layers = 4 projection\_hidden\_act = 'relu' layer\_norm\_eps = 1e-05 initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **window\_size** (`int`, *optional*, defaults to 8) —
  Image size of the spectrogram
* **num\_mel\_bins** (`int`, *optional*, defaults to 64) —
  Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
* **spec\_size** (`int`, *optional*, defaults to 256) —
  Desired input size of the spectrogram that the model supports. It can be different from the output of the
  `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
  of the audio models.
* **hidden\_act** (`str`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **patch\_size** (`int`, *optional*, defaults to 4) —
  Patch size for the audio spectrogram
* **patch\_stride** (`list`, *optional*, defaults to `[4, 4]`) —
  Patch stride for the audio spectrogram
* **num\_classes** (`int`, *optional*, defaults to 527) —
  Number of classes used for the head training
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer’s
  output,which is sent to the projection MLP layer.
* **projection\_dim** (`int`, *optional*, defaults to 512) —
  Hidden size of the projection layer.
* **depths** (`list`, *optional*, defaults to `[2, 2, 6, 2]`) —
  Depths used for the Swin Layers of the audio model
* **num\_attention\_heads** (`list`, *optional*, defaults to `[4, 8, 16, 32]`) —
  Number of attention heads used for the Swin Layers of the audio model
* **enable\_fusion** (`bool`, *optional*, defaults to `False`) —
  Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
  best results.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the encoder.
* **fusion\_type** (`[type]`, *optional*) —
  Fusion type used for the patch fusion.
* **patch\_embed\_input\_channels** (`int`, *optional*, defaults to 1) —
  Number of channels used for the input spectrogram
* **flatten\_patch\_embeds** (`bool`, *optional*, defaults to `True`) —
  Whether or not to flatten the patch embeddings
* **patch\_embeds\_hidden\_size** (`int`, *optional*, defaults to 96) —
  Hidden size of the patch embeddings. It is used as the number of output channels.
* **enable\_patch\_layer\_norm** (`bool`, *optional*, defaults to `True`) —
  Whether or not to enable layer normalization for the patch embeddings
* **drop\_path\_rate** (`float`, *optional*, defaults to 0.0) —
  Drop path rate for the patch fusion
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **qkv\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add a bias to the query, key, value projections.
* **mlp\_ratio** (`float`, *optional*, defaults to 4.0) —
  Ratio of the mlp hidden dim to embedding dim.
* **aff\_block\_r** (`int`, *optional*, defaults to 4) —
  downsize\_ratio used in the AudioFF block
* **num\_hidden\_layers** (`int`, *optional*, defaults to 4) —
  Number of hidden layers in the Transformer encoder.
* **projection\_hidden\_act** (`str`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **layer\_norm\_eps** (`[type]`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
  testing).

This is the configuration class to store the configuration of a [ClapAudioModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModel). It is used to instantiate a
CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP
[laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ClapAudioConfig, ClapAudioModel

>>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
>>> configuration = ClapAudioConfig()

>>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
>>> model = ClapAudioModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ClapFeatureExtractor

### class transformers.ClapFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/feature_extraction_clap.py#L34)

( feature\_size = 64 sampling\_rate = 48000 hop\_length = 480 max\_length\_s = 10 fft\_window\_size = 1024 padding\_value = 0.0 return\_attention\_mask = False frequency\_min: float = 0 frequency\_max: float = 14000 top\_db: typing.Optional[int] = None truncation: str = 'fusion' padding: str = 'repeatpad' \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 64) —
  The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters
  (`n_mels`).
* **sampling\_rate** (`int`, *optional*, defaults to 48000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves
  to warn users if the audio fed to the feature extractor does not have the same sampling rate.
* **hop\_length** (`int`,*optional*, defaults to 480) —
  Length of the overlapping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split
  in smaller `frames` with a step of `hop_length` between each frame.
* **max\_length\_s** (`int`, *optional*, defaults to 10) —
  The maximum input length of the model in seconds. This is used to pad the audio.
* **fft\_window\_size** (`int`, *optional*, defaults to 1024) —
  Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency
  resolution of the spectrogram. 400 means that the fourier transform is computed on windows of 400 samples.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  Padding value used to pad the audio. Should correspond to silences.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should return the attention masks corresponding to the input.
* **frequency\_min** (`float`, *optional*, defaults to 0) —
  The lowest frequency of interest. The STFT will not be computed for values below this.
* **frequency\_max** (`float`, *optional*, defaults to 14000) —
  The highest frequency of interest. The STFT will not be computed for values above this.
* **top\_db** (`float`, *optional*) —
  The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the
  `audio_utils.power_to_db` function
* **truncation** (`str`, *optional*, defaults to `"fusion"`) —
  Truncation pattern for long audio inputs. Two patterns are available:
  + `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a
    downsampled version of the entire mel spectrogram.
    If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy
    of the original mel obtained from the padded audio.
  + `rand_trunc` will select a random crop of the mel spectrogram.
* **padding** (`str`, *optional*, defaults to `"repeatpad"`) —
  Padding pattern for shorter audio inputs. Three patterns were originally implemented:
  + `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
  + `repeat`: the audio is repeated and then cut to fit the `max_length`
  + `pad`: the audio is padded.

Constructs a CLAP feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the *Short Time
Fourier Transform* (STFT) which should match pytorch’s `torch.stft` equivalent.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/feature_extraction_clap.py#L139)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance, except for the
mel filter banks, which do not need to be saved or printed as they are too long.

Serializes this instance to a Python dictionary.

## ClapProcessor

### class transformers.ClapProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/processing_clap.py#L23)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** ([ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor)) —
  The audio processor is a required input.
* **tokenizer** ([RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast)) —
  The tokenizer is a required input.

Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

[ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) offers all the functionalities of [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) and [RobertaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizerFast). See the
`__call__()` and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

## ClapModel

### class transformers.ClapModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1579)

( config: ClapConfig  )

Parameters

* **config** ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Clap Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1712)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None is\_longer: typing.Optional[torch.BoolTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.clap.modeling_clap.ClapOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details ([ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) uses
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) for processing audios).
* **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*) —
  Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
  the features.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clap.modeling_clap.ClapOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for audio-text similarity.
* **logits\_per\_audio** (`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`) — The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`) — The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel).
* **audio\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The audio embeddings obtained by applying the projection layer to the pooled output of [ClapAudioModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModel).
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The output of the [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel).
* **audio\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.audio_model_output`, defaults to `None`) — The output of the [ClapAudioModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModel).

The [ClapModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from datasets import load_dataset
>>> from transformers import AutoProcessor, ClapModel

>>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

>>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

>>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
>>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
```

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1614)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by
applying the projection layer to the pooled output of [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel).

Examples:


```
>>> from transformers import AutoTokenizer, ClapModel

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

>>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_audio\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1662)

( input\_features: typing.Optional[torch.Tensor] = None is\_longer: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → audio\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details ([ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) uses
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) for processing audios).
* **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*) —
  Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
  the features.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

audio\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The audio embeddings obtained by
applying the projection layer to the pooled output of [ClapAudioModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModel).

Examples:


```
>>> from transformers import AutoFeatureExtractor, ClapModel
>>> import torch

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
>>> random_audio = torch.rand((16_000))
>>> inputs = feature_extractor(random_audio, return_tensors="pt")
>>> audio_features = model.get_audio_features(**inputs)
```

## ClapTextModel

### class transformers.ClapTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1477)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
cross-attention is added between the self-attention layers, following the architecture described in *Attention is
all you need*\_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
`add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

.. \_*Attention is all you need*: <https://huggingface.co/papers/1706.03762>

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1502)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None token\_type\_ids: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None inputs\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.

The [ClapTextModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## ClapTextModelWithProjection

### class transformers.ClapTextModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1811)

( config: ClapTextConfig  )

Parameters

* **config** ([ClapTextConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Clap Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1827)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.clap.modeling_clap.ClapTextModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clap.modeling_clap.ClapTextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapTextModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) and inputs.

* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ClapTextModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapTextModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from transformers import AutoTokenizer, ClapTextModelWithProjection

>>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
>>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

>>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
```

## ClapAudioModel

### class transformers.ClapAudioModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1404)

( config: ClapAudioConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1417)

( input\_features: typing.Optional[torch.FloatTensor] = None is\_longer: typing.Optional[torch.BoolTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details ([ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) uses
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) for processing audios).
* **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*) —
  Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
  the features.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ClapAudioModel](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from datasets import load_dataset
>>> from transformers import AutoProcessor, ClapAudioModel

>>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
>>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

>>> inputs = processor(audios=audio_sample, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## ClapAudioModelWithProjection

### class transformers.ClapAudioModelWithProjection

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1876)

( config: ClapAudioConfig  )

Parameters

* **config** ([ClapAudioConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Clap Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clap/modeling_clap.py#L1890)

( input\_features: typing.Optional[torch.FloatTensor] = None is\_longer: typing.Optional[torch.BoolTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → `transformers.models.clap.modeling_clap.ClapAudioModelOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details ([ClapProcessor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapProcessor) uses
  [ClapFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapFeatureExtractor) for processing audios).
* **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*) —
  Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
  the features.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clap.modeling_clap.ClapAudioModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapAudioModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClapConfig](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapConfig)) and inputs.

* **audio\_embeds** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — The Audio embeddings obtained by applying the projection layer to the pooler\_output.
* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*, defaults to `None`) — Sequence of hidden-states at the output of the last layer of the model.
* **hidden\_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [ClapAudioModelWithProjection](/docs/transformers/v4.56.2/en/model_doc/clap#transformers.ClapAudioModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> from datasets import load_dataset
>>> from transformers import ClapAudioModelWithProjection, ClapProcessor

>>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
>>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

>>> dataset = load_dataset("hf-internal-testing/ashraq-esc50-1-dog-example")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> inputs = processor(audios=audio_sample, return_tensors="pt")
>>> outputs = model(**inputs)
>>> audio_embeds = outputs.audio_embeds
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/clap.md)
