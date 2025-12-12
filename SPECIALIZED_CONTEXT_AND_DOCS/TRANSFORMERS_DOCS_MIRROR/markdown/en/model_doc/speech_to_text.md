*This model was released on 2020-10-11 and added to Hugging Face Transformers on 2021-03-10.*

# Speech2Text

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Speech2Text model was proposed in [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://huggingface.co/papers/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It’s a
transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech
Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are
fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the
transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST:
[LibriSpeech](http://www.openslr.org/12), [CoVoST 2](https://github.com/facebookresearch/covost), [MuST-C](https://ict.fbk.eu/must-c/).

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text).

## Inference

Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech
signal. It’s a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The
`generate()` method can be used for inference.

The [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) class is responsible for extracting the log-mel filter-bank
features. The [Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) wraps [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) and
[Speech2TextTokenizer](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer) into a single instance to both extract the input features and decode the
predicted token ids.

The feature extractor depends on `torchaudio` and the tokenizer depends on `sentencepiece` so be sure to
install those packages before running the examples. You could either install those as extra speech dependencies with
`pip install transformers"[speech, sentencepiece]"` or install the packages separately with `pip install torchaudio sentencepiece`. Also `torchaudio` requires the development version of the [libsndfile](http://www.mega-nerd.com/libsndfile/) package which can be installed via a system package manager. On Ubuntu it can
be installed as follows: `apt install libsndfile1-dev`

* ASR and Speech Translation


```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

* Multilingual speech translation

  For multilingual speech translation models, `eos_token_id` is used as the `decoder_start_token_id` and
  the target language id is forced as the first generated token. To force the target language id as the first
  generated token, pass the `forced_bos_token_id` parameter to the `generate()` method. The following
  example shows how to translate English speech to French text using the *facebook/s2t-medium-mustc-multilingual-st*
  checkpoint.


```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

See the [model hub](https://huggingface.co/models?filter=speech_to_text) to look for Speech2Text checkpoints.

## Speech2TextConfig

### class transformers.Speech2TextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/configuration_speech_to_text.py#L24)

( vocab\_size = 10000 encoder\_layers = 12 encoder\_ffn\_dim = 2048 encoder\_attention\_heads = 4 decoder\_layers = 6 decoder\_ffn\_dim = 2048 decoder\_attention\_heads = 4 encoder\_layerdrop = 0.0 decoder\_layerdrop = 0.0 use\_cache = True is\_encoder\_decoder = True activation\_function = 'relu' d\_model = 256 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 decoder\_start\_token\_id = 2 scale\_embedding = True pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 max\_source\_positions = 6000 max\_target\_positions = 1024 num\_conv\_layers = 2 conv\_kernel\_sizes = (5, 5) conv\_channels = 1024 input\_feat\_per\_channel = 80 input\_channels = 1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 10000) —
  Vocabulary size of the Speech2Text model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [Speech2TextModel](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextModel)
* **encoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of encoder layers.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in encoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 4) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of decoder layers.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 2048) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 4) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556) for
  more details.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.0) —
  The LayerDrop probability for the decoder. See the [LayerDrop paper](https://huggingface.co/papers/1909.11556) for
  more details.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether the model should return the last key/values attentions (not used by all models).
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is set up as an encoder-decoder architecture for sequence-to-sequence tasks.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"silu"` and `"gelu_new"` are supported.
* **d\_model** (`int`, *optional*, defaults to 256) —
  Dimensionality of the layers and the pooler layer.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **init\_std** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **decoder\_start\_token\_id** (`int`, *optional*, defaults to 2) —
  The initial token ID of the decoder when decoding sequences.
* **scale\_embedding** (`bool`, *optional*, defaults to `True`) —
  Whether the embeddings are scaled by the square root of `d_model`.
* **pad\_token\_id** (`int`, *optional*, defaults to 1) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the beginning-of-sequence token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the end-of-sequence token.
* **max\_source\_positions** (`int`, *optional*, defaults to 6000) —
  The maximum sequence length of log-mel filter-bank features that this model might ever be used with.
* **max\_target\_positions** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically, set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **num\_conv\_layers** (`int`, *optional*, defaults to 2) —
  Number of 1D convolutional layers in the conv module.
* **conv\_kernel\_sizes** (`tuple[int]`, *optional*, defaults to `(5, 5)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the conv module. The length
  of `conv_kernel_sizes` has to match `num_conv_layers`.
* **conv\_channels** (`int`, *optional*, defaults to 1024) —
  An integer defining the number of output channels of each convolution layers except the final one in the
  conv module.
* **input\_feat\_per\_channel** (`int`, *optional*, defaults to 80) —
  An integer specifying the size of feature vector. This is also the dimensions of log-mel filter-bank
  features.
* **input\_channels** (`int`, *optional*, defaults to 1) —
  An integer specifying number of input channels of the input feature vector.

This is the configuration class to store the configuration of a [Speech2TextModel](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextModel). It is used to instantiate a
Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Speech2Text
[facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Speech2TextConfig, Speech2TextModel

>>> # Initializing a Speech2Text s2t_transformer_s style configuration
>>> configuration = Speech2TextConfig()

>>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
>>> model = Speech2TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Speech2TextTokenizer

### class transformers.Speech2TextTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L50)

( vocab\_file spm\_file bos\_token = '<s>' eos\_token = '</s>' pad\_token = '<pad>' unk\_token = '<unk>' do\_upper\_case = False do\_lower\_case = False tgt\_lang = None lang\_codes = None additional\_special\_tokens = None sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **spm\_file** (`str`) —
  Path to the [SentencePiece](https://github.com/google/sentencepiece) model file
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sentence token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sentence token.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **do\_upper\_case** (`bool`, *optional*, defaults to `False`) —
  Whether or not to uppercase the output when decoding.
* **do\_lower\_case** (`bool`, *optional*, defaults to `False`) —
  Whether or not to lowercase the input when tokenizing.
* **tgt\_lang** (`str`, *optional*) —
  A string representing the target language.
* **sp\_model\_kwargs** (`dict`, *optional*) —
  Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
  SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
  to set:
  + `enable_sampling`: Enable subword regularization.
  + `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

    - `nbest_size = {0,1}`: No sampling is performed.
    - `nbest_size > 1`: samples from the nbest\_size results.
    - `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice)
      using forward-filtering-and-backward-sampling algorithm.
  + `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
    BPE-dropout.
* \***\*kwargs** —
  Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

Construct an Speech2Text tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L206)

( token\_ids\_0 token\_ids\_1 = None  )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L213)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3432)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) — The first tokenized sequence.
* **token\_ids\_1** (`list[int]`, *optional*) — The second tokenized sequence.

Returns

`list[int]`

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type
IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L257)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## Speech2TextFeatureExtractor

### class transformers.Speech2TextFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py#L36)

( feature\_size = 80 sampling\_rate = 16000 num\_mel\_bins = 80 padding\_value = 0.0 dither = 0.0 do\_ceptral\_normalize = True normalize\_means = True normalize\_vars = True \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 80) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **num\_mel\_bins** (`int`, *optional*, defaults to 80) —
  Number of Mel-frequency bins.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding vectors.
* **dither** (`float`, *optional*, defaults to 0.0) —
  Adds dithering. In other words, adds a small Gaussian noise to each frame.
  E.g. use 4.0 to add dithering with a normal distribution centered
  around 0.0 with standard deviation 4.0 (assuming [-32k,+32k] range of kaldi waveform).
  The value 0.0 means no dithering.
  Dithering has similar effect as `mel_floor`. It reduces the high log\_mel\_fbank
  values for signals with hard-zero sections, when VAD cutoff is present in the signal.
* **do\_ceptral\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
* **normalize\_means** (`bool`, *optional*, defaults to `True`) —
  Whether or not to zero-mean normalize the extracted features.
* **normalize\_vars** (`bool`, *optional*, defaults to `True`) —
  Whether or not to unit-variance normalize the extracted features.

Constructs a Speech2Text feature extractor.

This feature extractor inherits from [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using TorchAudio if installed or using numpy
otherwise, and applies utterance-level cepstral mean and variance normalization to the extracted features.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py#L177)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False max\_length: typing.Optional[int] = None truncation: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None return\_attention\_mask: typing.Optional[bool] = None \*\*kwargs  )

Parameters

* **raw\_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **truncation** (`bool`) —
  Activates truncation to cut input sequences longer than *max\_length* to *max\_length*.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature\_extractor’s default.

  [What are attention masks?](../glossary#attention-mask)

  For Speech2TextTransformer models, `attention_mask` should always be passed for batched inference, to
  avoid subtle bugs.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values / vectors.

Main method to featurize and prepare for the model one or several sequence(s).

## Speech2TextProcessor

### class transformers.Speech2TextProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/processing_speech_to_text.py#L25)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`Speech2TextFeatureExtractor`) —
  An instance of [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor). The feature extractor is a required input.
* **tokenizer** (`Speech2TextTokenizer`) —
  An instance of [Speech2TextTokenizer](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer). The tokenizer is a required input.

Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
single processor.

[Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) offers all the functionalities of [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) and
[Speech2TextTokenizer](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer). See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more
information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/processing_speech_to_text.py#L49)

( \*args \*\*kwargs  )

When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor’s
[**call**()](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__) and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to Speech2TextTokenizer’s
[**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two methods for more
information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1272)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike] cache\_dir: typing.Union[str, os.PathLike, NoneType] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained feature\_extractor hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a feature extractor file saved using the
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved feature extractor JSON *file*, e.g.,
    `./my_model_directory/preprocessor_config.json`.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor
[ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer
`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the
methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L653)

( save\_directory push\_to\_hub: bool = False legacy\_serialization: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
  be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **legacy\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to save processor attributes in separate config files (legacy) or in processor’s config
  file as a nested dict. Saving all attributes in a single dict will become the default in future versions.
  Set to `legacy_serialization=True` until then.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it
can be reloaded using the [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and
[save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the
methods above for more information.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1419)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode). Please
refer to the docstring of this method for more information.

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/processing_utils.py#L1428)

( \*args \*\*kwargs  )

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.decode). Please refer to
the docstring of this method for more information.

## Speech2TextModel

### class transformers.Speech2TextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1044)

( config: Speech2TextConfig  )

Parameters

* **config** ([Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Speech To Text Text Model outputting raw hidden-states without any specific head on to.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1063)

( input\_features: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.LongTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor). See [Speech2TextFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__) for details ([Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) uses
  [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) for processing audios).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `SpeechToTextTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  SpeechToText uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read
  `modeling_speech_to_text._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the
  paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
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

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [Speech2TextModel](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import Speech2TextModel, AutoFeatureExtractor
>>> from datasets import load_dataset

>>> model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 256]
```

## Speech2TextForConditionalGeneration

### class transformers.Speech2TextForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1196)

( config: Speech2TextConfig  )

Parameters

* **config** ([Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Speech2Text Model with a language modeling head. Can be used for summarization.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1214)

( input\_features: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.Tensor] = None decoder\_head\_mask: typing.Optional[torch.Tensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.LongTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor). See [Speech2TextFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__) for details ([Speech2TextProcessor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) uses
  [Speech2TextFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) for processing audios).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using `SpeechToTextTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  SpeechToText uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read
  `modeling_speech_to_text._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the
  paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
  Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
  or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
  only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Speech2TextConfig](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [Speech2TextForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(inputs=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/speech_to_text.md)
