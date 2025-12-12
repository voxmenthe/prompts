*This model was released on 2021-10-14 and added to Hugging Face Transformers on 2023-02-03.*

# SpeechT5

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The SpeechT5 model was proposed in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://huggingface.co/papers/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

The abstract from the paper is the following:

*Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder. Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.*

This model was contributed by [Matthijs](https://huggingface.co/Matthijs). The original code can be found [here](https://github.com/microsoft/SpeechT5).

## SpeechT5Config

### class transformers.SpeechT5Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/configuration_speecht5.py#L27)

( vocab\_size = 81 hidden\_size = 768 encoder\_layers = 12 encoder\_attention\_heads = 12 encoder\_ffn\_dim = 3072 encoder\_layerdrop = 0.1 decoder\_layers = 6 decoder\_ffn\_dim = 3072 decoder\_attention\_heads = 12 decoder\_layerdrop = 0.1 hidden\_act = 'gelu' positional\_dropout = 0.1 hidden\_dropout = 0.1 attention\_dropout = 0.1 activation\_dropout = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 scale\_embedding = False feat\_extract\_norm = 'group' feat\_proj\_dropout = 0.0 feat\_extract\_activation = 'gelu' conv\_dim = (512, 512, 512, 512, 512, 512, 512) conv\_stride = (5, 2, 2, 2, 2, 2, 2) conv\_kernel = (10, 3, 3, 3, 3, 2, 2) conv\_bias = False num\_conv\_pos\_embeddings = 128 num\_conv\_pos\_embedding\_groups = 16 apply\_spec\_augment = True mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 decoder\_start\_token\_id = 2 num\_mel\_bins = 80 speech\_decoder\_prenet\_layers = 2 speech\_decoder\_prenet\_units = 256 speech\_decoder\_prenet\_dropout = 0.5 speaker\_embedding\_dim = 512 speech\_decoder\_postnet\_layers = 5 speech\_decoder\_postnet\_units = 256 speech\_decoder\_postnet\_kernel = 5 speech\_decoder\_postnet\_dropout = 0.5 reduction\_factor = 2 max\_speech\_positions = 4000 max\_text\_positions = 450 encoder\_max\_relative\_position = 160 use\_guided\_attention\_loss = True guided\_attention\_loss\_num\_heads = 2 guided\_attention\_loss\_sigma = 0.4 guided\_attention\_loss\_scale = 10.0 use\_cache = True is\_encoder\_decoder = True \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 81) —
  Vocabulary size of the SpeechT5 model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed to the forward method of [SpeechT5Model](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **encoder\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer decoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer decoder.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability for the decoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **positional\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the text position encoding layers.
* **hidden\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-5) —
  The epsilon used by the layer normalization layers.
* **scale\_embedding** (`bool`, *optional*, defaults to `False`) —
  Scale embeddings by diving by sqrt(d\_model).
* **feat\_extract\_norm** (`str`, *optional*, defaults to `"group"`) —
  The norm to be applied to 1D convolutional layers in the speech encoder pre-net. One of `"group"` for group
  normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
  convolutional layers.
* **feat\_proj\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for output of the speech encoder pre-net.
* **feat\_extract\_activation** (`str,` optional`, defaults to` “gelu”`) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string,` “gelu”`,` “relu”`,` “selu”`and`“gelu\_new”` are supported.
* **conv\_dim** (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) —
  A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
  speech encoder pre-net. The length of *conv\_dim* defines the number of 1D convolutional layers.
* **conv\_stride** (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the speech encoder pre-net. The
  length of *conv\_stride* defines the number of convolutional layers and has to match the length of
  *conv\_dim*.
* **conv\_kernel** (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the speech encoder pre-net.
  The length of *conv\_kernel* defines the number of convolutional layers and has to match the length of
  *conv\_dim*.
* **conv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether the 1D convolutional layers have a bias.
* **num\_conv\_pos\_embeddings** (`int`, *optional*, defaults to 128) —
  Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
  embeddings layer.
* **num\_conv\_pos\_embedding\_groups** (`int`, *optional*, defaults to 16) —
  Number of groups of 1D convolutional positional embeddings layer.
* **apply\_spec\_augment** (`bool`, *optional*, defaults to `True`) —
  Whether to apply *SpecAugment* data augmentation to the outputs of the speech encoder pre-net. For
  reference see [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
  Recognition](https://huggingface.co/papers/1904.08779).
* **mask\_time\_prob** (`float`, *optional*, defaults to 0.05) —
  Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
  procedure generates ”mask\_time\_prob*len(time\_axis)/mask\_time\_length” independent masks over the axis. If
  reasoning from the probability of each feature vector to be chosen as the start of the vector span to be
  masked,* mask\_time\_prob *should be `prob\_vector\_start*mask\_time\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is True`.
* **mask\_time\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the time axis.
* **mask\_time\_min\_masks** (`int`, *optional*, defaults to 2), —
  The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
  irrespectively of `mask_feature_prob`. Only relevant if ”mask\_time\_prob\*len(time\_axis)/mask\_time\_length <
  mask\_time\_min\_masks”
* **mask\_feature\_prob** (`float`, *optional*, defaults to 0.0) —
  Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
  masking procedure generates ”mask\_feature\_prob*len(feature\_axis)/mask\_time\_length” independent masks over
  the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
  span to be masked,* mask\_feature\_prob *should be `prob\_vector\_start*mask\_feature\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is
  True`.
* **mask\_feature\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the feature axis.
* **mask\_feature\_min\_masks** (`int`, *optional*, defaults to 0), —
  The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
  step, irrespectively of `mask_feature_prob`. Only relevant if
  ”mask\_feature\_prob\*len(feature\_axis)/mask\_feature\_length < mask\_feature\_min\_masks”
* **num\_mel\_bins** (`int`, *optional*, defaults to 80) —
  Number of mel features used per input features. Used by the speech decoder pre-net. Should correspond to
  the value used in the [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) class.
* **speech\_decoder\_prenet\_layers** (`int`, *optional*, defaults to 2) —
  Number of layers in the speech decoder pre-net.
* **speech\_decoder\_prenet\_units** (`int`, *optional*, defaults to 256) —
  Dimensionality of the layers in the speech decoder pre-net.
* **speech\_decoder\_prenet\_dropout** (`float`, *optional*, defaults to 0.5) —
  The dropout probability for the speech decoder pre-net layers.
* **speaker\_embedding\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of the *XVector* embedding vectors.
* **speech\_decoder\_postnet\_layers** (`int`, *optional*, defaults to 5) —
  Number of layers in the speech decoder post-net.
* **speech\_decoder\_postnet\_units** (`int`, *optional*, defaults to 256) —
  Dimensionality of the layers in the speech decoder post-net.
* **speech\_decoder\_postnet\_kernel** (`int`, *optional*, defaults to 5) —
  Number of convolutional filter channels in the speech decoder post-net.
* **speech\_decoder\_postnet\_dropout** (`float`, *optional*, defaults to 0.5) —
  The dropout probability for the speech decoder post-net layers.
* **reduction\_factor** (`int`, *optional*, defaults to 2) —
  Spectrogram length reduction factor for the speech decoder inputs.
* **max\_speech\_positions** (`int`, *optional*, defaults to 4000) —
  The maximum sequence length of speech features that this model might ever be used with.
* **max\_text\_positions** (`int`, *optional*, defaults to 450) —
  The maximum sequence length of text features that this model might ever be used with.
* **encoder\_max\_relative\_position** (`int`, *optional*, defaults to 160) —
  Maximum distance for relative position embedding in the encoder.
* **use\_guided\_attention\_loss** (`bool`, *optional*, defaults to `True`) —
  Whether to apply guided attention loss while training the TTS model.
* **guided\_attention\_loss\_num\_heads** (`int`, *optional*, defaults to 2) —
  Number of attention heads the guided attention loss will be applied to. Use -1 to apply this loss to all
  attention heads.
* **guided\_attention\_loss\_sigma** (`float`, *optional*, defaults to 0.4) —
  Standard deviation for guided attention loss.
* **guided\_attention\_loss\_scale** (`float`, *optional*, defaults to 10.0) —
  Scaling coefficient for guided attention loss (also known as lambda).
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [SpeechT5Model](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model). It is used to instantiate a
SpeechT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the SpeechT5
[microsoft/speecht5\_asr](https://huggingface.co/microsoft/speecht5_asr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SpeechT5Model, SpeechT5Config

>>> # Initializing a "microsoft/speecht5_asr" style configuration
>>> configuration = SpeechT5Config()

>>> # Initializing a model (with random weights) from the "microsoft/speecht5_asr" style configuration
>>> model = SpeechT5Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SpeechT5HifiGanConfig

### class transformers.SpeechT5HifiGanConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/configuration_speecht5.py#L340)

( model\_in\_dim = 80 sampling\_rate = 16000 upsample\_initial\_channel = 512 upsample\_rates = [4, 4, 4, 4] upsample\_kernel\_sizes = [8, 8, 8, 8] resblock\_kernel\_sizes = [3, 7, 11] resblock\_dilation\_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]] initializer\_range = 0.01 leaky\_relu\_slope = 0.1 normalize\_before = True \*\*kwargs  )

Parameters

* **model\_in\_dim** (`int`, *optional*, defaults to 80) —
  The number of frequency bins in the input log-mel spectrogram.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
* **upsample\_initial\_channel** (`int`, *optional*, defaults to 512) —
  The number of input channels into the upsampling network.
* **upsample\_rates** (`tuple[int]` or `list[int]`, *optional*, defaults to `[4, 4, 4, 4]`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the upsampling network. The
  length of *upsample\_rates* defines the number of convolutional layers and has to match the length of
  *upsample\_kernel\_sizes*.
* **upsample\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 8, 8]`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the upsampling network. The
  length of *upsample\_kernel\_sizes* defines the number of convolutional layers and has to match the length of
  *upsample\_rates*.
* **resblock\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`) —
  A tuple of integers defining the kernel sizes of the 1D convolutional layers in the multi-receptive field
  fusion (MRF) module.
* **resblock\_dilation\_sizes** (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) —
  A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
  multi-receptive field fusion (MRF) module.
* **initializer\_range** (`float`, *optional*, defaults to 0.01) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **leaky\_relu\_slope** (`float`, *optional*, defaults to 0.1) —
  The angle of the negative slope used by the leaky ReLU activation.
* **normalize\_before** (`bool`, *optional*, defaults to `True`) —
  Whether or not to normalize the spectrogram before vocoding using the vocoder’s learned mean and variance.

This is the configuration class to store the configuration of a `SpeechT5HifiGanModel`. It is used to instantiate
a SpeechT5 HiFi-GAN vocoder model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5
[microsoft/speecht5\_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

>>> # Initializing a "microsoft/speecht5_hifigan" style configuration
>>> configuration = SpeechT5HifiGanConfig()

>>> # Initializing a model (with random weights) from the "microsoft/speecht5_hifigan" style configuration
>>> model = SpeechT5HifiGan(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SpeechT5Tokenizer

### class transformers.SpeechT5Tokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/tokenization_speecht5.py#L35)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' normalize = False sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
  contains the vocabulary necessary to instantiate a tokenizer.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The begin of sequence token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **normalize** (`bool`, *optional*, defaults to `False`) —
  Whether to convert numeric quantities in the text to their spelt-out english counterparts.
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
* **sp\_model** (`SentencePieceProcessor`) —
  The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L2828)

( text: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None add\_special\_tokens: bool = True padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False truncation: typing.Union[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy, NoneType] = None max\_length: typing.Optional[int] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None padding\_side: typing.Optional[str] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None return\_token\_type\_ids: typing.Optional[bool] = None return\_attention\_mask: typing.Optional[bool] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **add\_special\_tokens** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
  Activates and controls padding. Accepts the following values:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) —
  Activates and controls truncation. Accepts the following values:
  + `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  + `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  + `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
* **max\_length** (`int`, *optional*) —
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
* **stride** (`int`, *optional*, defaults to 0) —
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **padding\_side** (`str`, *optional*) —
  The side on which the model should have padding applied. Should be selected between [‘right’, ‘left’].
  Default value is picked from the class attribute of the same name.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **return\_token\_type\_ids** (`bool`, *optional*) —
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer’s default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
* **return\_overflowing\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
* **return\_special\_tokens\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return special tokens mask information.
* **return\_offsets\_mapping** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using
  Python’s tokenizer, this method will raise `NotImplementedError`.
* **return\_length** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the lengths of the encoded inputs.
* **verbose** (`bool`, *optional*, defaults to `True`) —
  Whether or not to print more information and warnings.
* \***\*kwargs** — passed to the `self.tokenize()` method

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)
* **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *“token\_type\_ids”* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)
* **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
* **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
* **length** — The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/tokenization_speecht5.py#L205)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3867)

( token\_ids: typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `str`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str`

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/tokenization_utils_base.py#L3833)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None \*\*kwargs  ) → `list[str]`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces. If `None`, will default to
  `self.clean_up_tokenization_spaces`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]`

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

## SpeechT5FeatureExtractor

### class transformers.SpeechT5FeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/feature_extraction_speecht5.py#L31)

( feature\_size: int = 1 sampling\_rate: int = 16000 padding\_value: float = 0.0 do\_normalize: bool = False num\_mel\_bins: int = 80 hop\_length: int = 16 win\_length: int = 64 win\_function: str = 'hann\_window' frame\_signal\_scale: float = 1.0 fmin: float = 80 fmax: float = 7600 mel\_floor: float = 1e-10 reduction\_factor: int = 2 return\_attention\_mask: bool = True \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values.
* **do\_normalize** (`bool`, *optional*, defaults to `False`) —
  Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
  improve the performance for some models.
* **num\_mel\_bins** (`int`, *optional*, defaults to 80) —
  The number of mel-frequency bins in the extracted spectrogram features.
* **hop\_length** (`int`, *optional*, defaults to 16) —
  Number of ms between windows. Otherwise referred to as “shift” in many papers.
* **win\_length** (`int`, *optional*, defaults to 64) —
  Number of ms per window.
* **win\_function** (`str`, *optional*, defaults to `"hann_window"`) —
  Name for the window function used for windowing, must be accessible via `torch.{win_function}`
* **frame\_signal\_scale** (`float`, *optional*, defaults to 1.0) —
  Constant multiplied in creating the frames before applying DFT. This argument is deprecated.
* **fmin** (`float`, *optional*, defaults to 80) —
  Minimum mel frequency in Hz.
* **fmax** (`float`, *optional*, defaults to 7600) —
  Maximum mel frequency in Hz.
* **mel\_floor** (`float`, *optional*, defaults to 1e-10) —
  Minimum value of mel frequency banks.
* **reduction\_factor** (`int`, *optional*, defaults to 2) —
  Spectrogram length reduction factor. This argument is deprecated.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `True`) —
  Whether or not [**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor.__call__) should return `attention_mask`.

Constructs a SpeechT5 feature extractor.

This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by
the SpeechT5 speech encoder prenet.

This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder
prenet.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/feature_extraction_speecht5.py#L180)

( audio: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]], NoneType] = None audio\_target: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]], NoneType] = None padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False max\_length: typing.Optional[int] = None truncation: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **audio** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*) —
  The sequence or batch of sequences to be processed. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. This outputs waveform features. Must
  be mono channel audio, not stereo, i.e. single float per timestep.
* **audio\_target** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`, *optional*) —
  The sequence or batch of sequences to be processed as targets. Each sequence can be a numpy array, a
  list of float values, a list of numpy arrays or a list of list of float values. This outputs log-mel
  spectrogram features.
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) —
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
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `audio` or `audio_target` input was sampled. It is strongly recommended
  to pass `sampling_rate` at the forward call to prevent silent errors.

Main method to featurize and prepare for the model one or several sequence(s).

Pass in a value for `audio` to extract waveform features. Pass in a value for `audio_target` to extract log-mel
spectrogram features.

## SpeechT5Processor

### class transformers.SpeechT5Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L20)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`SpeechT5FeatureExtractor`) —
  An instance of [SpeechT5FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor). The feature extractor is a required input.
* **tokenizer** (`SpeechT5Tokenizer`) —
  An instance of [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). The tokenizer is a required input.

Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.

[SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) offers all the functionalities of [SpeechT5FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor) and [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See
the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L40)

( \*args \*\*kwargs  )

Processes audio and text input, as well as audio and text targets.

You can process audio by using the argument `audio`, or process audio targets by using the argument
`audio_target`. This forwards the arguments to SpeechT5FeatureExtractor’s
[**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor.__call__).

You can process text by using the argument `text`, or process text labels by using the argument `text_target`.
This forwards the arguments to SpeechT5Tokenizer’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__).

Valid input combinations are:

* `text` only
* `audio` only
* `text_target` only
* `audio_target` only
* `text` and `audio_target`
* `audio` and `audio_target`
* `text` and `text_target`
* `audio` and `text_target`

Please refer to the docstring of the above two methods for more information.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/processing_speecht5.py#L111)

( \*args \*\*kwargs  )

Collates the audio and text inputs, as well as their targets, into a padded batch.

Audio inputs are padded by SpeechT5FeatureExtractor’s [pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad). Text inputs are padded
by SpeechT5Tokenizer’s [pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad).

Valid input combinations are:

* `input_ids` only
* `input_values` only
* `labels` only, either log-mel spectrograms or text tokens
* `input_ids` and log-mel spectrogram `labels`
* `input_values` and text `labels`

Please refer to the docstring of the above two methods for more information.

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

## SpeechT5Model

### class transformers.SpeechT5Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L1948)

( config: SpeechT5Config encoder: typing.Optional[torch.nn.modules.module.Module] = None decoder: typing.Optional[torch.nn.modules.module.Module] = None  )

Parameters

* **config** ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **encoder** (`PreTrainedModel`, *optional*) —
  The encoder model to use.
* **decoder** (`PreTrainedModel`, *optional*) —
  The decoder model to use.

The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L1993)

( input\_values: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_values: typing.Optional[torch.Tensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None use\_cache: typing.Optional[bool] = None speaker\_embeddings: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`) —
  Depending on which encoder is being used, the `input_values` are either: float values of the input raw
  speech waveform, or indices of input sequence tokens in the vocabulary, or hidden states.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_values** (`torch.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Depending on which decoder is being used, the `decoder_input_values` are either: float values of log-mel
  filterbank features extracted from the raw speech waveform, or indices of decoder input sequence tokens in
  the vocabulary, or hidden states.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
  also be used by default.

  If you want to change padding behavior, you should read `SpeechT5Decoder._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
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
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **speaker\_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*) —
  Tensor containing the speaker embeddings.
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

[transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [SpeechT5Model](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## SpeechT5ForSpeechToText

### class transformers.SpeechT5ForSpeechToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2109)

( config: SpeechT5Config  )

Parameters

* **config** ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a speech encoder and a text decoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2151)

( input\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) should be used for padding
  and conversion into a tensor of type `torch.FloatTensor`. See [SpeechT5Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) for details.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  SpeechT5 uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
  also be used by default.

  If you want to change padding behavior, you should read `SpeechT5Decoder._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
  or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
  only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

  Label indices can be obtained using [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) and inputs.

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

The [SpeechT5ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
>>> from datasets import load_dataset

>>> dataset = load_dataset(
...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
... )  # doctest: +IGNORE_RESULT
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
>>> model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

>>> # audio file is decoded on the fly
>>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> predicted_ids = model.generate(**inputs, max_length=100)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
>>> transcription[0]
'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
```


```
>>> inputs["labels"] = processor(text_target=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
19.68
```

## SpeechT5ForTextToSpeech

### class transformers.SpeechT5ForTextToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2439)

( config: SpeechT5Config  )

Parameters

* **config** ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a text encoder and a speech decoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2475)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_values: typing.Optional[torch.FloatTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None speaker\_embeddings: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.FloatTensor] = None stop\_labels: typing.Optional[torch.Tensor] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See [encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`) —
  Float values of input mel spectrogram.

  SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
  also be used by default.

  If you want to change padding behavior, you should read `SpeechT5Decoder._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
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
* **speaker\_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*) —
  Tensor containing the speaker embeddings.
* **labels** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*) —
  Float values of target mel spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
  computation. Spectrograms can be obtained using [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor). See [SpeechT5Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__)
  for details.
* **stop\_labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Binary tensor indicating the position of the stop token in the sequence.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Spectrogram generation loss.
* **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) — The predicted spectrogram.
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

The [SpeechT5ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForTextToSpeech) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
>>> import torch

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
>>> model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
>>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
>>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

>>> set_seed(555)  # make deterministic

>>> # generate speech
>>> speech = model.generate(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
>>> speech.shape
torch.Size([15872])
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2610)

( input\_ids: LongTensor attention\_mask: typing.Optional[torch.LongTensor] = None speaker\_embeddings: typing.Optional[torch.FloatTensor] = None threshold: float = 0.5 minlenratio: float = 0.0 maxlenratio: float = 20.0 vocoder: typing.Optional[torch.nn.modules.module.Module] = None output\_cross\_attentions: bool = False return\_output\_lengths: bool = False \*\*kwargs  ) → `tuple(torch.FloatTensor)` comprising various elements depending on the inputs

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SpeechT5Tokenizer](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See [encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Attention mask from the tokenizer, required for batched inference to signal to the model where to
  ignore padded tokens from the input\_ids.
* **speaker\_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*) —
  Tensor containing the speaker embeddings.
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The generated sequence ends when the predicted stop token probability exceeds this value.
* **minlenratio** (`float`, *optional*, defaults to 0.0) —
  Used to calculate the minimum required length for the output sequence.
* **maxlenratio** (`float`, *optional*, defaults to 20.0) —
  Used to calculate the maximum allowed length for the output sequence.
* **vocoder** (`nn.Module`, *optional*) —
  The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
  spectrogram.
* **output\_cross\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the attentions tensors of the decoder’s cross-attention layers.
* **return\_output\_lengths** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the concrete spectrogram/waveform lengths.

Returns

`tuple(torch.FloatTensor)` comprising various elements depending on the inputs

* when `return_output_lengths` is False
  + **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
    `(output_sequence_length, config.num_mel_bins)` — The predicted log-mel spectrogram.
  + **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
    `(num_frames,)` — The predicted speech waveform.
  + **cross\_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
    `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)` — The outputs of the decoder’s cross-attention layers.
* when `return_output_lengths` is True
  + **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
    `(batch_size, output_sequence_length, config.num_mel_bins)` — The predicted log-mel spectrograms that
    are padded to the maximum length.
  + **spectrogram\_lengths** (*optional*, returned when no `vocoder` is provided) `list[Int]` — A list of
    all the concrete lengths for each spectrogram.
  + **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
    `(batch_size, num_frames)` — The predicted speech waveforms that are padded to the maximum length.
  + **waveform\_lengths** (*optional*, returned when a `vocoder` is provided) `list[Int]` — A list of all
    the concrete lengths for each waveform.
  + **cross\_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
    `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)` — The outputs of the decoder’s cross-attention layers.

Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
speech waveform using a vocoder.

## SpeechT5ForSpeechToSpeech

### class transformers.SpeechT5ForSpeechToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2804)

( config: SpeechT5Config  )

Parameters

* **config** ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a speech encoder and a speech decoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2830)

( input\_values: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None decoder\_input\_values: typing.Optional[torch.FloatTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.FloatTensor] = None decoder\_head\_mask: typing.Optional[torch.FloatTensor] = None cross\_attn\_head\_mask: typing.Optional[torch.Tensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None speaker\_embeddings: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.FloatTensor] = None stop\_labels: typing.Optional[torch.Tensor] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) should be used for padding and conversion into
  a tensor of type `torch.FloatTensor`. See [SpeechT5Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) for details.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`) —
  Float values of input mel spectrogram.

  SpeechT5 uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
  `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
  `past_key_values`).
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
  also be used by default.

  If you want to change padding behavior, you should read `SpeechT5Decoder._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **decoder\_head\_mask** (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*) —
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
* **speaker\_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*) —
  Tensor containing the speaker embeddings.
* **labels** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*) —
  Float values of target mel spectrogram. Spectrograms can be obtained using [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor). See
  [SpeechT5Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) for details.
* **stop\_labels** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Binary tensor indicating the position of the stop token in the sequence.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SpeechT5Config](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Spectrogram generation loss.
* **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) — The predicted spectrogram.
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

The [SpeechT5ForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToSpeech) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset(
...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
... )  # doctest: +IGNORE_RESULT
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
>>> model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
>>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

>>> # audio file is decoded on the fly
>>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

>>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

>>> set_seed(555)  # make deterministic

>>> # generate speech
>>> speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
>>> speech.shape
torch.Size([77824])
```

#### generate\_speech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L2960)

( input\_values: FloatTensor speaker\_embeddings: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None threshold: float = 0.5 minlenratio: float = 0.0 maxlenratio: float = 20.0 vocoder: typing.Optional[torch.nn.modules.module.Module] = None output\_cross\_attentions: bool = False return\_output\_lengths: bool = False  ) → `tuple(torch.FloatTensor)` comprising various elements depending on the inputs

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform.

  Values can be obtained by loading a *.flac* or *.wav* audio file into an array of type `list[float]`,
  a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`)
  or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [SpeechT5Processor](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor) should be used for padding and
  conversion into a tensor of type `torch.FloatTensor`. See [SpeechT5Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) for details.
* **speaker\_embeddings** (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*) —
  Tensor containing the speaker embeddings.
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing convolution and attention on padding token indices. Mask values selected in
  `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **threshold** (`float`, *optional*, defaults to 0.5) —
  The generated sequence ends when the predicted stop token probability exceeds this value.
* **minlenratio** (`float`, *optional*, defaults to 0.0) —
  Used to calculate the minimum required length for the output sequence.
* **maxlenratio** (`float`, *optional*, defaults to 20.0) —
  Used to calculate the maximum allowed length for the output sequence.
* **vocoder** (`nn.Module`, *optional*, defaults to `None`) —
  The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
  spectrogram.
* **output\_cross\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the attentions tensors of the decoder’s cross-attention layers.
* **return\_output\_lengths** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return the concrete spectrogram/waveform lengths.

Returns

`tuple(torch.FloatTensor)` comprising various elements depending on the inputs

* when `return_output_lengths` is False
  + **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
    `(output_sequence_length, config.num_mel_bins)` — The predicted log-mel spectrogram.
  + **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
    `(num_frames,)` — The predicted speech waveform.
  + **cross\_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
    `torch.FloatTensor` of shape `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)` — The outputs of the decoder’s cross-attention layers.
* when `return_output_lengths` is True
  + **spectrograms** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
    `(batch_size, output_sequence_length, config.num_mel_bins)` — The predicted log-mel spectrograms that
    are padded to the maximum length.
  + **spectrogram\_lengths** (*optional*, returned when no `vocoder` is provided) `list[Int]` — A list of
    all the concrete lengths for each spectrogram.
  + **waveforms** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
    `(batch_size, num_frames)` — The predicted speech waveforms that are padded to the maximum length.
  + **waveform\_lengths** (*optional*, returned when a `vocoder` is provided) `list[Int]` — A list of all
    the concrete lengths for each waveform.
  + **cross\_attentions** (*optional*, returned when `output_cross_attentions` is `True`)
    `torch.FloatTensor` of shape `(batch_size, config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)` — The outputs of the decoder’s cross-attention layers.

Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a
speech waveform using a vocoder.

## SpeechT5HifiGan

### class transformers.SpeechT5HifiGan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L3118)

( config: SpeechT5HifiGanConfig  )

Parameters

* **config** ([SpeechT5HifiGanConfig](/docs/transformers/v4.56.2/en/model_doc/speecht5#transformers.SpeechT5HifiGanConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

HiFi-GAN vocoder.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/speecht5/modeling_speecht5.py#L3187)

( spectrogram: FloatTensor  ) → `torch.FloatTensor`

Parameters

* **spectrogram** (`torch.FloatTensor`) —
  Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length, config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

Returns

`torch.FloatTensor`

Tensor containing the speech waveform. If the input spectrogram is batched, will be of
shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.

Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
waveform.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/speecht5.md)
