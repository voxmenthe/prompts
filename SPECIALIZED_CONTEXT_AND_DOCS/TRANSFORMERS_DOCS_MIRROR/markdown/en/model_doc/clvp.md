*This model was released on 2023-05-12 and added to Hugging Face Transformers on 2023-11-10.*

# CLVP

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The CLVP (Contrastive Language-Voice Pretrained Transformer) model was proposed in [Better speech synthesis through scaling](https://huggingface.co/papers/2305.07243) by James Betker.

The abstract from the paper is the following:

*In recent years, the field of image generation has been revolutionized by the application of autoregressive transformers and DDPMs. These approaches model the process of image generation as a step-wise probabilistic processes and leverage large amounts of compute and data to learn the image distribution. This methodology of improving performance need not be confined to images. This paper describes a way to apply advances in the image generative domain to speech synthesis. The result is TorToise - an expressive, multi-voice text-to-speech system.*

This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/neonbjb/tortoise-tts).

## Usage tips

1. CLVP is an integral part of the Tortoise TTS model.
2. CLVP can be used to compare different generated speech candidates with the provided text, and the best speech tokens are forwarded to the diffusion model.
3. The use of the `ClvpModelForConditionalGeneration.generate()` method is strongly recommended for tortoise usage.
4. Note that the CLVP model expects the audio to be sampled at 22.05 kHz contrary to other audio models which expects 16 kHz.

## Brief Explanation:

* The [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer) tokenizes the text input, and the [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor) extracts the log mel-spectrogram from the desired audio.
* `ClvpConditioningEncoder` takes those text tokens and audio representations and converts them into embeddings conditioned on the text and audio.
* The [ClvpForCausalLM](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpForCausalLM) uses those embeddings to generate multiple speech candidates.
* Each speech candidate is passed through the speech encoder ([ClvpEncoder](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpEncoder)) which converts them into a vector representation, and the text encoder ([ClvpEncoder](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpEncoder)) converts the text tokens into the same latent space.
* At the end, we compare each speech vector with the text vector to see which speech vector is most similar to the text vector.
* `ClvpModelForConditionalGeneration.generate()` compresses all of the logic described above into a single method.

Example :


```
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library).
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> sample = ds[0]["audio"]

>>> # Define processor and model.
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output.
>>> processor_output = processor(raw_speech=sample["array"], sampling_rate=sample["sampling_rate"], text=text, return_tensors="pt")
>>> generated_output = model.generate(**processor_output)
```

## ClvpConfig

### class transformers.ClvpConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/configuration_clvp.py#L316)

( text\_config = None speech\_config = None decoder\_config = None projection\_dim = 768 logit\_scale\_init\_value = 2.6592 initializer\_factor = 1.0 \*\*kwargs  )

Parameters

* **text\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize the CLVP text encoder.
* **speech\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize CLVP speech encoder.
* **decoder\_config** (`dict`, *optional*) —
  Dictionary of configuration options used to initialize [ClvpDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpDecoderConfig).
* **projection\_dim** (`int`, *optional*, defaults to 768) —
  Dimensionality of text and speech projection layers.
* **logit\_scale\_init\_value** (`float`, *optional*, defaults to 2.6592) —
  The initial value of the *logit\_scale* parameter. Default is used as per the original CLVP implementation.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
  testing).
* **kwargs** (*optional*) —
  Dictionary of keyword arguments.

[ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig) is the configuration class to store the configuration of a [ClvpModelForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpModelForConditionalGeneration). It
is used to instantiate a CLVP model according to the specified arguments, defining the text model, speech model and
decoder model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
of the CLVP [susnato/clvp\_dev](https://huggingface.co/susnato/clvp_dev) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ClvpConfig, ClvpModelForConditionalGeneration

>>> # Initializing a ClvpConfig with susnato/clvp_dev style configuration
>>> configuration = ClvpConfig()

>>> # Initializing a ClvpModelForConditionalGeneration (with random weights) from the susnato/clvp_dev style configuration
>>> model = ClvpModelForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

>>> # We can also initialize a CLVPConfig from a CLVPTextConfig, CLVPSpeechConfig and a CLVPAutoRegressiveConfig
>>> from transformers import ClvpEncoderConfig, ClvpDecoderConfig

>>> # Initializing a CLVP text, CLVP speech and CLVP decoder configuration
>>> config_text = ClvpEncoderConfig()
>>> config_speech = ClvpEncoderConfig()
>>> decoder_config = ClvpDecoderConfig()

>>> config = ClvpConfig.from_sub_model_configs(config_text, config_speech, decoder_config)
```

#### from\_sub\_model\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/configuration_clvp.py#L407)

( text\_config: ClvpEncoderConfig speech\_config: ClvpEncoderConfig decoder\_config: ClvpDecoderConfig \*\*kwargs  ) → [ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)

Parameters

* **text\_config** (`ClvpEncoderConfig`) —
  Text model configuration of type [ClvpEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpEncoderConfig).
* **speech\_config** (`ClvpEncoderConfig`) —
  Speech model configuration of type [ClvpEncoderConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpEncoderConfig).
* **decoder\_config** (`ClvpDecoderConfig`) —
  Decoder model configuration of type [ClvpDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpDecoderConfig).

Returns

[ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)

An instance of a configuration object

Instantiate a [ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig) (or a derived class) from CLVP text model configuration, CLVP speech model
configuration and CLVP decoder model configuration.

## ClvpEncoderConfig

### class transformers.ClvpEncoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/configuration_clvp.py#L27)

( vocab\_size = 256 hidden\_size = 768 intermediate\_size = 1536 projection\_dim = 768 num\_hidden\_layers = 20 num\_attention\_heads = 12 hidden\_act = 'gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.1 dropout = 0.1 use\_rotary\_embedding = True use\_attention\_bias = False summary\_type = 'mean' initializer\_factor = 1.0 bos\_token\_id = 255 eos\_token\_id = 0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 256) —
  Vocabulary size of the CLVP Encoder model.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **intermediate\_size** (`int`, *optional*, defaults to 1536) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **projection\_dim** (`int`, *optional*, defaults to 768) —
  Dimensionality of the projection vector.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 20) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the feed-forward layers in `ClvpEncoderMLP`.
* **use\_rotary\_embedding** (`bool`, *optional*, defaults to `True`) —
  Whether to use rotary\_embedding or not.
* **use\_attention\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether to use bias in Query, Key and Value layers during self attention.
* **summary\_type** (`str`, *optional*, defaults to `"mean"`) —
  What strategy to use to get pooler\_output from the last\_hidden\_state. `"last"`, `"first"`, `"mean"` and
  `"cls_index"` are supported.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
  testing).
* **bos\_token\_id** (`int`, *optional*, defaults to 255) —
  Beginning of sequence token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 0) —
  End of sequence token id.

This is the configuration class to store the configuration of a [ClvpEncoder](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpEncoder). It is used to instantiate a CLVP
text or CLVP speech encoder according to the specified arguments. Instantiating a configuration with the defaults
will yield a similar configuration to that of the encoder of the CLVP
[susnato/clvp\_dev](https://huggingface.co/susnato/clvp_dev) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import ClvpEncoderConfig, ClvpEncoder

>>> # Initializing a ClvpEncoderConfig with susnato/clvp_dev style configuration
>>> encoder_configuration = ClvpEncoderConfig()

>>> # Initializing a ClvpEncoder (with random weights) from the susnato/clvp_dev style configuration
>>> model = ClvpEncoder(encoder_configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ClvpDecoderConfig

### class transformers.ClvpDecoderConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/configuration_clvp.py#L159)

( vocab\_size = 8194 max\_position\_embeddings = 608 max\_text\_tokens = 404 hidden\_size = 1024 num\_hidden\_layers = 30 num\_attention\_heads = 16 n\_inner = None num\_mel\_attn\_blocks = 6 activation\_function = 'gelu\_new' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attention\_dropout = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 summary\_type = 'cls\_index' summary\_use\_proj = True summary\_activation = None summary\_proj\_to\_labels = True summary\_first\_dropout = 0.1 use\_cache = True bos\_token\_id = 8192 eos\_token\_id = 8193 feature\_size = 80 use\_attention\_bias = True initializer\_factor = 1.0 decoder\_fixing\_codes = [83, 45, 45, 248] \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 8194) —
  Vocabulary size of the model.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 608) —
  The maximum sequence length of mel tokens that this model might ever be used with. Similar to `n_positions`
  in `GPT2Config`.
* **max\_text\_tokens** (`int`, *optional*, defaults to 404) —
  The maximum sequence length of text tokens that this model might ever be used with. Similar to
  `n_positions` in `GPT2Config`.
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the embeddings and hidden states.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 30) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **n\_inner** (`int`, *optional*) —
  Dimensionality of the inner feed-forward layers. `None` will set it to 4 times `hidden_size`.
* **num\_mel\_attn\_blocks** (`int`, *optional*, defaults to 6) —
  Denotes the number of self attention layers in `ClvpConditioningEncoder`.
* **activation\_function** (`str`, *optional*, defaults to `"gelu_new"`) —
  Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
* **resid\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **embd\_pdrop** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the embeddings.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention.
* **layer\_norm\_epsilon** (`float`, *optional*, defaults to 1e-05) —
  The epsilon to use in the layer normalization layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **summary\_type** (`string`, *optional*, defaults to `"cls_index"`) —
  Argument used when doing sequence summary.

  Has to be one of the following options:

  + `"last"`: Take the last token hidden state (like XLNet).
  + `"first"`: Take the first token hidden state (like BERT).
  + `"mean"`: Take the mean of all tokens hidden states.
  + `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
  + `"attn"`: Not implemented now, use multi-head attention.
* **summary\_use\_proj** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add a projection after the vector extraction.
* **summary\_activation** (`str`, *optional*) —
  Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
* **summary\_proj\_to\_labels** (`bool`, *optional*, defaults to `True`) —
  Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
* **summary\_first\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio to be used after the projection and activation.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **bos\_token\_id** (`int`, *optional*, defaults to 8192) —
  Beginning of sequence token id, used at the start of the generation.
* **eos\_token\_id** (`int`, *optional*, defaults to 8193) —
  End of sequence token id, used in the method
  `ClvpModelForConditionalGeneration.fix_speech_decoder_output()` to correct decoder outputs.
* **feature\_size** (`int`, *optional*, defaults to 80) —
  The feature dimension of the extracted mel features. This value is used in `ClvpConditioningEncoder`.
* **use\_attention\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in Query, Key and Value layers during self attention.
* **initializer\_factor** (`float`, *optional*, defaults to 1.0) —
  A factor for initializing all weight matrices (should be kept to 1.0, used internally for initialization
  testing).
* **decoder\_fixing\_codes** (`list`, *optional*, defaults to `[83, 45, 45, 248]`) —
  These values are used in the method `fix_speech_decoder_output` to fix decoder generated outputs.

This is the configuration class to store the configuration of a [ClvpDecoder](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpDecoder). It is used to instantiate a CLVP
Decoder Model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Decoder part of the CLVP
[susnato/clvp\_dev](https://huggingface.co/susnato/clvp_dev) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

The architecture is similar to GPT2.

Example:


```
>>> from transformers import ClvpDecoderConfig, ClvpDecoder

>>> # Initializing a ClvpDecoderConfig with susnato/clvp_dev style configuration
>>> decoder_configuration = ClvpDecoderConfig()

>>> # Initializing a ClvpDecoder (with random weights) from the susnato/clvp_dev style configuration
>>> model = ClvpDecoder(decoder_configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## ClvpTokenizer

### class transformers.ClvpTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/tokenization_clvp.py#L78)

( vocab\_file merges\_file errors = 'replace' unk\_token = '[UNK]' bos\_token = '<|endoftext|>' eos\_token = '[STOP]' pad\_token = '[STOP]' add\_prefix\_space = False add\_bos\_token = False add\_eos\_token = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **merges\_file** (`str`) —
  Path to the merges file.
* **errors** (`str`, *optional*, defaults to `"replace"`) —
  Paradigm to follow when decoding bytes to UTF-8. See
  [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
* **unk\_token** (`str`, *optional*, defaults to `"[UNK]"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **bos\_token** (`str`, *optional*, defaults to `"<|endoftext|>"`) —
  The beginning of sequence token.
* **eos\_token** (`str`, *optional*, defaults to `"[STOP]"`) —
  The end of sequence token.
* **pad\_token** (`str`, *optional*, defaults to `"[STOP]"`) —
  The pad token of the sequence.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `False`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word. (CLVP tokenizer detect beginning of words by the preceding space).
* **add\_bos\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to add `bos_token` in front of the sequence when add\_special\_tokens=True.
* **add\_eos\_token** (`bool`, *optional*, defaults to `False`) —
  Whether to add `eos_token` in end of the sequence when add\_special\_tokens=True.

Construct a CLVP tokenizer. Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:


```
>>> from transformers import ClvpTokenizer

>>> tokenizer = ClvpTokenizer.from_pretrained("susnato/clvp_dev")
>>> tokenizer("Hello world")["input_ids"]
[62, 84, 28, 2, 179, 79]

>>> tokenizer(" Hello world")["input_ids"]
[2, 62, 84, 28, 2, 179, 79]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/tokenization_clvp.py#L337)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## ClvpFeatureExtractor

### class transformers.ClvpFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/feature_extraction_clvp.py#L33)

( feature\_size = 80 sampling\_rate = 22050 default\_audio\_length = 6 hop\_length = 256 chunk\_length = 30 n\_fft = 1024 padding\_value = 0.0 mel\_norms = None return\_attention\_mask = False \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 80) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 22050) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **default\_audio\_length** (`int`, *optional*, defaults to 6) —
  The default length of raw audio in seconds. If `max_length` is not set during `__call__` then it will
  automatically be set to default\_audio\_length \* `self.sampling_rate`.
* **hop\_length** (`int`, *optional*, defaults to 256) —
  Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
* **chunk\_length** (`int`, *optional*, defaults to 30) —
  The maximum number of chunks of `sampling_rate` samples used to trim and pad longer or shorter audio
  sequences.
* **n\_fft** (`int`, *optional*, defaults to 1024) —
  Size of the Fourier transform.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  Padding value used to pad the audio. Should correspond to silences.
* **mel\_norms** (`list` of length `feature_size`, *optional*) —
  If `mel_norms` is provided then it will be used to normalize the log-mel spectrograms along each
  mel-filter.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether to return the attention mask. If left to the default, it will return the attention mask.

  [What are attention masks?](../glossary#attention-mask)

Constructs a CLVP feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts log-mel-spectrogram features from raw speech using a custom numpy implementation of the `Short Time Fourier Transform` which should match pytorch’s `torch.stft` equivalent.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/feature_extraction_clvp.py#L131)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] sampling\_rate: typing.Optional[int] = None truncation: bool = True pad\_to\_multiple\_of: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_attention\_mask: typing.Optional[bool] = True padding: typing.Optional[str] = 'max\_length' max\_length: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **raw\_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors and allow automatic speech recognition
  pipeline.
* **truncation** (`bool`, *optional*, default to `True`) —
  Activates truncation to cut input sequences longer than *max\_length* to *max\_length*.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `True`) —
  Whether to return the attention mask. If left to the default, it will return the attention mask.

  [What are attention masks?](../glossary#attention-mask)
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values / vectors.
* **max\_length** (`int`, *optional*) —
  The maximum input length of the inputs.

`ClvpFeatureExtractor` is used to extract various voice specific properties such as the pitch and tone of the
voice, speaking speed, and even speaking defects like a lisp or stuttering from a sample voice or `raw_speech`.

First the voice is padded or truncated in a way such that it becomes a waveform of `self.default_audio_length`
seconds long and then the log-mel spectrogram is extracted from it.

## ClvpProcessor

### class transformers.ClvpProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/processing_clvp.py#L23)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`ClvpFeatureExtractor`) —
  An instance of [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor). The feature extractor is a required input.
* **tokenizer** (`ClvpTokenizer`) —
  An instance of [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer). The tokenizer is a required input.

Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

[ClvpProcessor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpProcessor) offers all the functionalities of [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor) and [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer). See the
[**call**()](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpProcessor.__call__), [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) and [batch\_decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.batch_decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/processing_clvp.py#L48)

( \*args \*\*kwargs  )

Forwards the `audio` and `sampling_rate` arguments to [**call**()](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor.__call__) and the `text`
argument to [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two methods for more
information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

## ClvpModelForConditionalGeneration

### class transformers.ClvpModelForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1440)

( config: ClvpConfig  )

Parameters

* **config** ([ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The composite CLVP model with a text encoder, speech encoder and speech decoder model.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1660)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None conditioning\_encoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None text\_encoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None return\_loss: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = False return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → `transformers.models.clvp.modeling_clvp.ClvpOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor). See [ClvpFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor.__call__) for details ([ClvpProcessor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpProcessor) uses
  [ClvpFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpFeatureExtractor) for processing audios).
* **conditioning\_encoder\_inputs\_embeds** (`torch.FloatTensor`, *optional*) —
  inputs\_embeds for `ClvpConditioningEncoder`. Can be used in place of `input_ids`.
* **text\_encoder\_inputs\_embeds** (`torch.FloatTensor`, *optional*) —
  inputs\_embeds for the text encoder model passed in place of `input_ids`.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **return\_loss** (`bool`, *optional*) —
  Whether or not to return the contrastive loss.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **output\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

`transformers.models.clvp.modeling_clvp.ClvpOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clvp.modeling_clvp.ClvpOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`) — Contrastive loss for speech-text similarity.
* **speech\_ids** (`torch.LongTensor`, *optional*) — speech\_ids (or speech candidates) generated by the `ClvpForCausalLM` model.
* **logits\_per\_speech** (`torch.FloatTensor` of shape `(speech_batch_size, text_batch_size)`) — The scaled dot product scores between `speech_embeds` and `text_embeds`. This represents the speech-text
  similarity scores.
* **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, speech_batch_size)`) — The scaled dot product scores between `text_embeds` and `speech_embeds`. This represents the text-speech
  similarity scores.
* **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of the text encoder
  model.
* **speech\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim`) — The speech embeddings obtained by applying the projection layer to the pooled output of the speech encoder
  model.
* **text\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.text_model_output`, defaults to `None`) — The pooled output of the `last_hidden_state` of the text encoder Model.
* **speech\_model\_output** (`<class '~modeling_outputs.BaseModelOutputWithPooling'>.speech_model_output`, defaults to `None`) — The pooled output of the `last_hidden_state` of the speech encoder Model.
* **decoder\_hidden\_states** (`torch.FloatTensor`, *optional*) — The hidden states of the decoder model.
* **text\_encoder\_hidden\_states** (`torch.FloatTensor`, *optional*) — The hidden states of the text encoder model.
* **speech\_encoder\_hidden\_states** (`torch.FloatTensor`, *optional*) — The hidden states of the speech encoder model.

The [ClvpModelForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpModelForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Examples:


```
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
>>> text = "This is an example text."

>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> audio = ds.sort("id")["audio"][0]
>>> audio_sample, sr = audio["array"], audio["sampling_rate"]

>>> # Define processor and model
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # processor outputs and model outputs
>>> processor_output = processor(raw_speech=audio_sample, sampling_rate=sr, text=text, return_tensors="pt")
>>> outputs = model(
...     input_ids=processor_output["input_ids"],
...     input_features=processor_output["input_features"],
...     return_dict=True,
... )
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1799)

( input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None pad\_to\_max\_mel\_tokens: typing.Optional[int] = None output\_hidden\_states: typing.Optional[bool] = None \*\*kwargs  ) → `ClvpOutput` or tuple

Parameters

* **input\_ids** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Input text Tokens. Processed from the [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **generation\_config** (`~generation.GenerationConfig`, *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which had the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **pad\_to\_max\_mel\_tokens** (`int`, *optional*) —
  Pads generated speech\_ids to the specified value. This is to implement the same logic from the official
  repo, link: <https://github.com/neonbjb/tortoise-tts/blob/80f89987a5abda5e2b082618cd74f9c7411141dc/tortoise/api.py#L430>
  and to make sure the logits are same.
  This does not affect generation quality so please don’t consider using it since it is less efficient.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of decoder model, text encoder and speech encoder models.

Returns

`ClvpOutput` or tuple

A `ClvpOutput` (if `return_dict_in_generate=True` or when
`config.return_dict_in_generate=True`) or a tuple.

Generate method for `ClvpModelForConditionalGeneration`, this method calls the `generate` method of
`ClvpForCausalLM` and then uses those generated `speech_ids` to process `text_embeds` and `speech_embeds` using
`ClvpEncoder`.

#### get\_text\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1508)

( input\_ids: typing.Optional[torch.LongTensor] = None text\_encoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None  ) → `torch.FloatTensor` of shape `(batch_size, output_dim)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
  provide it.

  [What are input IDs?](../glossary#input-ids)
* **text\_encoder\_inputs\_embeds** (`torch.FloatTensor`, *optional*) —
  inputs\_embeds for the text encoder model passed in place of `input_ids`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)

Returns

`torch.FloatTensor` of shape `(batch_size, output_dim)`

The text embeddings obtained by applying the projection layer to the pooled output of the CLVP Text
Model.

This method can be used to extract text\_embeds from a text. The text embeddings obtained by applying the
projection layer to the pooled output of the CLVP text encoder model.

Examples:


```
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text
>>> text = "This is an example text."

>>> # Define processor and model
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and text embeds
>>> processor_output = processor(text=text, return_tensors="pt")
>>> text_embeds = model.get_text_features(input_ids=processor_output["input_ids"])
```

#### get\_speech\_features

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1565)

( speech\_ids: typing.Optional[torch.LongTensor] = None input\_ids: typing.Optional[torch.LongTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None conditioning\_encoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None generation\_config: typing.Optional[transformers.generation.configuration\_utils.GenerationConfig] = None \*\*kwargs  ) → `torch.FloatTensor` of shape `(batch_size, output_dim)`

Parameters

* **speech\_ids** (`torch.LongTensor` of shape `(batch_size, num_speech_ids)`, *optional*) —
  Speech Tokens. Padding will be ignored by default should you provide it. If speech\_ids are provided
  then input\_ids and input\_features will be automatically ignored.
* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Input text Tokens. Processed from the [ClvpTokenizer](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpTokenizer). If speech\_ids is not provided, then input\_ids
  and input\_features will be used.
* **conditioning\_encoder\_inputs\_embeds** (`torch.FloatTensor`, *optional*) —
  inputs\_embeds for `ClvpConditioningEncoder`. Can be used in place of `input_ids`.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding speech token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **generation\_config** (`GenerationConfig`, *optional*) —
  generation config to control the generation of speech\_ids if they are not provided.

Returns

`torch.FloatTensor` of shape `(batch_size, output_dim)`

The speech embeddings obtained by applying the projection layer to the pooled output of the CLVP Speech
Model.

This method can be used to extract speech\_embeds. The speech embeddings are obtained by applying the speech
model on speech\_ids. If speech\_ids is not present but both input\_ids and input\_features are given then the
decoder model will be used to first generate the speech\_ids and then applying the speech model.

Examples:


```
>>> import datasets
>>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

>>> # Define the Text and Load the Audio (We are taking an audio example from HuggingFace Hub using `datasets` library)
>>> text = "This is an example text."
>>> ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
>>> audio = ds.sort("id")["audio"][0]
>>> audio_sample, sr = audio["array"], audio["sampling_rate"]

>>> # Define processor and model
>>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
>>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

>>> # Generate processor output and model output
>>> processor_output = processor(raw_speech=audio_sample, sampling_rate=sr, text=text, return_tensors="pt")
>>> speech_embeds = model.get_speech_features(
...     input_ids=processor_output["input_ids"], input_features=processor_output["input_features"]
... )
```

## ClvpForCausalLM

### class transformers.ClvpForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1246)

( config  )

Parameters

* **config** ([ClvpForCausalLM](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CLVP decoder model with a language modelling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1360)

( input\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
  `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
  are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Cross attentions weights after the attention softmax, used to compute the weighted average in the
  cross-attention heads.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.

The [ClvpForCausalLM](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## ClvpModel

### class transformers.ClvpModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1175)

( config: ClvpDecoderConfig  )

Parameters

* **config** ([ClvpDecoderConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpDecoderConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Clvp Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1190)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [ClvpModel](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## ClvpEncoder

### class transformers.ClvpEncoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L831)

( config: ClvpConfig  )

Parameters

* **config** — ClvpConfig

Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
`ClvpEncoderLayer`.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L863)

( input\_ids: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  input embeddings for the model. This bypasses the model’s internal embedding lookup matrix.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor`, *optional*) —
  Denotes the position ids of `input_ids`.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under
  returned tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
  for more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

## ClvpDecoder

### class transformers.ClvpDecoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L987)

( config  )

Transformer decoder consisting of *config.num\_hidden\_layers* layers. Each layer is a `ClvpDecoderLayer`

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/clvp/modeling_clvp.py#L1024)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.FloatTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None past\_key\_values: typing.Optional[tuple[tuple[torch.Tensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **past\_key\_values** (`tuple[tuple[torch.Tensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([ClvpConfig](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.

The [ClvpDecoder](/docs/transformers/v4.56.2/en/model_doc/clvp#transformers.ClvpDecoder) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/clvp.md)
