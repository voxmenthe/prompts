*This model was released on 2020-06-20 and added to Hugging Face Transformers on 2021-02-02.*

# Wav2Vec2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

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

Note: Meta (FAIR) released a new version of [Wav2Vec2-BERT 2.0](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2-bert) - it’s pretrained on 4.5M hours of audio. We especially recommend using it for fine-tuning tasks, e.g. as per [this guide](https://huggingface.co/blog/fine-tune-w2v2-bert).

## Usage tips

* Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
* Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded
  using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

> [!NOTE]
> The `head_mask` argument is ignored when using all attention implementation other than “eager”. If you have a `head_mask` and want it to have effect, load the model with `XXXModel.from_pretrained(model_id, attn_implementation="eager")`

## Using Flash Attention 2

Flash Attention 2 is an faster, optimized version of the model.

### Installation

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features). If your hardware is not compatible with Flash Attention 2, you can still benefit from attention kernel optimisations through Better Transformer support covered [above](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:


```
pip install -U flash-attn --no-build-isolation
```

### Usage

To load a model using Flash Attention 2, we can pass the argument `attn_implementation="flash_attention_2"` to [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). We’ll also load the model in half-precision (e.g. `torch.float16`), since it results in almost no degradation to audio quality but significantly lower memory usage and faster inference:


```
>>> from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
...
```

### Expected speedups

Below is an expected speedup diagram comparing the pure inference time between the native implementation in transformers of the `facebook/wav2vec2-large-960h-lv60-self` model and the flash-attention-2 and sdpa (scale-dot-product-attention) versions. . We show the average speedup obtained on the `librispeech_asr` `clean` validation split:

![](https://huggingface.co/datasets/kamilakesbi/transformers_image_doc/resolve/main/data/Wav2Vec2_speedup.png)

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Wav2Vec2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

Audio Classification

* A notebook on how to [leverage a pretrained Wav2Vec2 model for emotion classification](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb). 🌎
* [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).
* [Audio classification task guide](../tasks/audio_classification)

Automatic Speech Recognition

* A blog post on [boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram).
* A blog post on how to [finetune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english).
* A blog post on [finetuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2).
* A notebook on how to [create YouTube captions from any video by transcribing audio with Wav2Vec2](https://colab.research.google.com/github/Muennighoff/ytclipcc/blob/main/wav2vec_youtube_captions.ipynb). 🌎
* [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by a notebook on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).
* [Automatic speech recognition task guide](../tasks/asr)

🚀 Deploy

* A blog post on how to deploy Wav2Vec2 for [Automatic Speech Recognition with Hugging Face’s Transformers & Amazon SageMaker](https://www.philschmid.de/automatic-speech-recognition-sagemaker).

## Wav2Vec2Config

### class transformers.Wav2Vec2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/configuration_wav2vec2.py#L27)

( vocab\_size = 32 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout = 0.1 activation\_dropout = 0.1 attention\_dropout = 0.1 feat\_proj\_dropout = 0.0 feat\_quantizer\_dropout = 0.0 final\_dropout = 0.1 layerdrop = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 feat\_extract\_norm = 'group' feat\_extract\_activation = 'gelu' conv\_dim = (512, 512, 512, 512, 512, 512, 512) conv\_stride = (5, 2, 2, 2, 2, 2, 2) conv\_kernel = (10, 3, 3, 3, 3, 2, 2) conv\_bias = False num\_conv\_pos\_embeddings = 128 num\_conv\_pos\_embedding\_groups = 16 do\_stable\_layer\_norm = False apply\_spec\_augment = True mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 num\_codevectors\_per\_group = 320 num\_codevector\_groups = 2 contrastive\_logits\_temperature = 0.1 num\_negatives = 100 codevector\_dim = 256 proj\_codevector\_dim = 256 diversity\_loss\_weight = 0.1 ctc\_loss\_reduction = 'sum' ctc\_zero\_infinity = False use\_weighted\_layer\_sum = False classifier\_proj\_size = 256 tdnn\_dim = (512, 512, 512, 512, 1500) tdnn\_kernel = (5, 3, 3, 1, 1) tdnn\_dilation = (1, 2, 3, 1, 1) xvector\_output\_dim = 512 pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 add\_adapter = False adapter\_kernel\_size = 3 adapter\_stride = 2 num\_adapter\_layers = 3 output\_hidden\_size = None adapter\_attn\_dim = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 32) —
  Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) or `TFWav2Vec2Model`. Vocabulary size of the
  model. Defines the different tokens that can be represented by the *inputs\_ids* passed to the forward
  method of [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **final\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the final projection layer of [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC).
* **layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>) for more
  details.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **feat\_extract\_norm** (`str`, *optional*, defaults to `"group"`) —
  The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
  normalization of only the first 1D convolutional layer or `"layer"` for layer normalization of all 1D
  convolutional layers.
* **feat\_proj\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for output of the feature encoder.
* **feat\_extract\_activation** (`str,` optional`, defaults to` “gelu”`) -- The non-linear activation function (function or string) in the 1D convolutional layers of the feature extractor. If string,` “gelu”`,` “relu”`,` “selu”`and`“gelu\_new”` are supported.
* **feat\_quantizer\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for quantized feature encoder states.
* **conv\_dim** (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) —
  A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
  feature encoder. The length of *conv\_dim* defines the number of 1D convolutional layers.
* **conv\_stride** (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
  of *conv\_stride* defines the number of convolutional layers and has to match the length of *conv\_dim*.
* **conv\_kernel** (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
  length of *conv\_kernel* defines the number of convolutional layers and has to match the length of
  *conv\_dim*.
* **conv\_bias** (`bool`, *optional*, defaults to `False`) —
  Whether the 1D convolutional layers have a bias.
* **num\_conv\_pos\_embeddings** (`int`, *optional*, defaults to 128) —
  Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
  embeddings layer.
* **num\_conv\_pos\_embedding\_groups** (`int`, *optional*, defaults to 16) —
  Number of groups of 1D convolutional positional embeddings layer.
* **do\_stable\_layer\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is False` corresponds to applying layer norm after the attention layer.
* **apply\_spec\_augment** (`bool`, *optional*, defaults to `True`) —
  Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
  [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
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
* **num\_codevectors\_per\_group** (`int`, *optional*, defaults to 320) —
  Number of entries in each quantization codebook (group).
* **num\_codevector\_groups** (`int`, *optional*, defaults to 2) —
  Number of codevector groups for product codevector quantization.
* **contrastive\_logits\_temperature** (`float`, *optional*, defaults to 0.1) —
  The temperature *kappa* in the contrastive loss.
* **feat\_quantizer\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for the output of the feature encoder that’s used by the quantizer.
* **num\_negatives** (`int`, *optional*, defaults to 100) —
  Number of negative samples for the contrastive loss.
* **codevector\_dim** (`int`, *optional*, defaults to 256) —
  Dimensionality of the quantized feature vectors.
* **proj\_codevector\_dim** (`int`, *optional*, defaults to 256) —
  Dimensionality of the final projection of both the quantized and the transformer features.
* **diversity\_loss\_weight** (`int`, *optional*, defaults to 0.1) —
  The weight of the codebook diversity loss component.
* **ctc\_loss\_reduction** (`str`, *optional*, defaults to `"sum"`) —
  Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
  instance of [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC).
* **ctc\_zero\_infinity** (`bool`, *optional*, defaults to `False`) —
  Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
  occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
  of [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC).
* **use\_weighted\_layer\_sum** (`bool`, *optional*, defaults to `False`) —
  Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
  instance of [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification).
* **classifier\_proj\_size** (`int`, *optional*, defaults to 256) —
  Dimensionality of the projection before token mean-pooling for classification.
* **tdnn\_dim** (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`) —
  A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
  module of the *XVector* model. The length of *tdnn\_dim* defines the number of *TDNN* layers.
* **tdnn\_kernel** (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
  *XVector* model. The length of *tdnn\_kernel* has to match the length of *tdnn\_dim*.
* **tdnn\_dilation** (`tuple[int]` or `list[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`) —
  A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
  *XVector* model. The length of *tdnn\_dilation* has to match the length of *tdnn\_dim*.
* **xvector\_output\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of the *XVector* embedding vectors.
* **add\_adapter** (`bool`, *optional*, defaults to `False`) —
  Whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder. Can be very useful for
  warm-starting Wav2Vec2 for SpeechEncoderDecoder models.
* **adapter\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adapter\_stride** (`int`, *optional*, defaults to 2) —
  Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **num\_adapter\_layers** (`int`, *optional*, defaults to 3) —
  Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is True`.
* **adapter\_attn\_dim** (`int`, *optional*) —
  Dimension of the attention adapter weights to be used in each attention block. An example of a model using
  attention adapters is [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).
* **output\_hidden\_size** (`int`, *optional*) —
  Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
  if `add_adapter is True`.

This is the configuration class to store the configuration of a [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model). It is used to instantiate an
Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
with the defaults will yield a similar configuration to that of the Wav2Vec2
[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Wav2Vec2Config, Wav2Vec2Model

>>> # Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
>>> configuration = Wav2Vec2Config()

>>> # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
>>> model = Wav2Vec2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Wav2Vec2CTCTokenizer

### class transformers.Wav2Vec2CTCTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L115)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' unk\_token = '<unk>' pad\_token = '<pad>' word\_delimiter\_token = '|' replace\_word\_delimiter\_char = ' ' do\_lower\_case = False target\_lang = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  File containing the vocabulary.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sentence token.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sentence token.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **word\_delimiter\_token** (`str`, *optional*, defaults to `"|"`) —
  The token used for defining the end of a word.
* **do\_lower\_case** (`bool`, *optional*, defaults to `False`) —
  Whether or not to accept lowercase input and lowercase the output when decoding.
* **target\_lang** (`str`, *optional*) —
  A target language the tokenizer should set by default. `target_lang` has to be defined for multi-lingual,
  nested vocabulary such as [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).
* \***\*kwargs** —
  Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

Constructs a Wav2Vec2CTC tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to
the superclass for more information regarding such methods.

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L633)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L528)

( token\_ids: typing.Union[int, list[int], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_char\_offsets: bool = False output\_word\_offsets: bool = False \*\*kwargs  ) → `str` or `Wav2Vec2CTCTokenizerOutput`

Parameters

* **token\_ids** (`Union[int, list[int], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces.
* **output\_char\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output character offsets. Character offsets can be used in combination with the
  sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

  Please take a look at the example below to better understand how to make use of `output_char_offsets`.
* **output\_word\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
  and model downsampling rate to compute the time-stamps of transcribed words.

  Please take a look at the example below to better understand how to make use of `output_word_offsets`.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`str` or `Wav2Vec2CTCTokenizerOutput`

The list of decoded
sentences. Will be a `Wav2Vec2CTCTokenizerOutput` when
`output_char_offsets == True` or `output_word_offsets == True`.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

Example:


```
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

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L458)

( sequences: typing.Union[list[int], list[list[int]], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: typing.Optional[bool] = None output\_char\_offsets: bool = False output\_word\_offsets: bool = False \*\*kwargs  ) → `list[str]` or `Wav2Vec2CTCTokenizerOutput`

Parameters

* **sequences** (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor, tf.Tensor]`) —
  List of tokenized input ids. Can be obtained using the `__call__` method.
* **skip\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not to remove special tokens in the decoding.
* **clean\_up\_tokenization\_spaces** (`bool`, *optional*) —
  Whether or not to clean up the tokenization spaces.
* **output\_char\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output character offsets. Character offsets can be used in combination with the
  sampling rate and model downsampling rate to compute the time-stamps of transcribed characters.

  Please take a look at the Example of [decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.decode) to better understand how to make
  use of `output_char_offsets`. [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.batch_decode) works the same way with batched
  output.
* **output\_word\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
  and model downsampling rate to compute the time-stamps of transcribed words.

  Please take a look at the Example of [decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.decode) to better understand how to make
  use of `output_word_offsets`. [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer.batch_decode) works the same way with batched
  output.
* **kwargs** (additional keyword arguments, *optional*) —
  Will be passed to the underlying model specific decode method.

Returns

`list[str]` or `Wav2Vec2CTCTokenizerOutput`

The list of decoded
sentences. Will be a `Wav2Vec2CTCTokenizerOutput` when
`output_char_offsets == True` or `output_word_offsets == True`.

Convert a list of lists of token ids into a list of strings by calling decode.

#### set\_target\_lang

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L198)

( target\_lang: str  )

Set the target language of a nested multi-lingual dictionary

## Wav2Vec2FeatureExtractor

### class transformers.Wav2Vec2FeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31)

( feature\_size = 1 sampling\_rate = 16000 padding\_value = 0.0 return\_attention\_mask = False do\_normalize = True \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 1) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding values.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
  improve the performance for some models, *e.g.*,
  [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
* **return\_attention\_mask** (`bool`, *optional*, defaults to `False`) —
  Whether or not [**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) should return `attention_mask`.

  Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
  [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
  `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask`
  should be passed.

  For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
  [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be
  passed for batched inference.

Constructs a Wav2Vec2 feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L102)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = False max\_length: typing.Optional[int] = None truncation: bool = False pad\_to\_multiple\_of: typing.Optional[int] = None return\_attention\_mask: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **raw\_speech** (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
  values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
  stereo, i.e. single float per timestep.
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

  Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
  [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
  `attention_mask`. For such models, `input_values` should simply be padded with 0 and no
  `attention_mask` should be passed.

  For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
  [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should
  be passed for batched inference.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —

Main method to featurize and prepare for the model one or several sequence(s).

## Wav2Vec2Processor

### class transformers.Wav2Vec2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/processing_wav2vec2.py#L33)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`Wav2Vec2FeatureExtractor`) —
  An instance of [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). The feature extractor is a required input.
* **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  An instance of [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). The tokenizer is a required input.

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
processor.

[Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) offers all the functionalities of [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
See the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/processing_wav2vec2.py#L75)

( audio: typing.Union[ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor']] = None text: typing.Union[str, list[str], NoneType] = None images = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.wav2vec2.processing\_wav2vec2.Wav2Vec2ProcessorKwargs]  )

Parameters

* **audio** (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*) —
  An audio input is passed to [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__).
* **text** (`str`, `List[str]`, *optional*) —
  A text input is passed to [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__).

This method forwards all arguments to [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and/or
[PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) depending on the input modality and returns their outputs. If both modalities are passed, [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) are called.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/processing_wav2vec2.py#L131)

( \*args \*\*kwargs  )

Parameters

* **input\_features** —
  When the first argument is a dictionary containing a batch of tensors, or the `input_features` argument is present, it is passed to [Wav2Vec2FeatureExtractor.pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad).
* **labels** —
  When the `label` argument is present, it is passed to [PreTrainedTokenizer.pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad).

This method operates on batches of extracted features and/or tokenized text. It forwards all arguments to
[Wav2Vec2FeatureExtractor.pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and/or [PreTrainedTokenizer.pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad) depending on the input modality and returns their outputs. If both modalities are passed, [Wav2Vec2FeatureExtractor.pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and [PreTrainedTokenizer.pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad) are called.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/processing_wav2vec2.py#L56)

( pretrained\_model\_name\_or\_path \*\*kwargs  )

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

## Wav2Vec2ProcessorWithLM

### class transformers.Wav2Vec2ProcessorWithLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L69)

( feature\_extractor: FeatureExtractionMixin tokenizer: PreTrainedTokenizerBase decoder: BeamSearchDecoderCTC  )

Parameters

* **feature\_extractor** ([Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) or [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor)) —
  An instance of [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) or [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor). The feature extractor is a required input.
* **tokenizer** ([Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer)) —
  An instance of [Wav2Vec2CTCTokenizer](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer). The tokenizer is a required input.
* **decoder** (`pyctcdecode.BeamSearchDecoderCTC`) —
  An instance of `pyctcdecode.BeamSearchDecoderCTC`. The decoder is a required input.

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
with language model support into a single processor for language model boosted speech recognition decoding.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L222)

( \*args \*\*kwargs  )

When used in normal mode, this method forwards all its arguments to the feature extractor’s
`__call__()` and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to
Wav2Vec2CTCTokenizer’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__). Please refer to the docstring of the above two
methods for more information.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L261)

( \*args \*\*kwargs  )

When used in normal mode, this method forwards all its arguments to the feature extractor’s
`~FeatureExtractionMixin.pad` and returns its output. If used in the context
`as_target_processor()` this method forwards all its arguments to
Wav2Vec2CTCTokenizer’s [pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad). Please refer to the docstring of the above two methods
for more information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L120)

( pretrained\_model\_name\_or\_path \*\*kwargs  )

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
  Additional keyword arguments passed along to both [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) and
  [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

Instantiate a [Wav2Vec2ProcessorWithLM](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM) from a pretrained Wav2Vec2 processor.

This class method is simply calling the feature extractor’s
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), Wav2Vec2CTCTokenizer’s
[from\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), and
`pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`.

Please refer to the docstrings of the methods above for more information.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L116)

( save\_directory  )

#### batch\_decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L292)

( logits: ndarray pool: typing.Optional[<bound method BaseContext.Pool of <multiprocessing.context.DefaultContext object at 0x7f39a8b14700>>] = None num\_processes: typing.Optional[int] = None beam\_width: typing.Optional[int] = None beam\_prune\_logp: typing.Optional[float] = None token\_min\_logp: typing.Optional[float] = None hotwords: typing.Optional[collections.abc.Iterable[str]] = None hotword\_weight: typing.Optional[float] = None alpha: typing.Optional[float] = None beta: typing.Optional[float] = None unk\_score\_offset: typing.Optional[float] = None lm\_score\_boundary: typing.Optional[bool] = None output\_word\_offsets: bool = False n\_best: int = 1  )

Parameters

* **logits** (`np.ndarray`) —
  The logits output vector of the model representing the log probabilities for each token.
* **pool** (`multiprocessing.Pool`, *optional*) —
  An optional user-managed pool. If not set, one will be automatically created and closed. The pool
  should be instantiated *after* `Wav2Vec2ProcessorWithLM`. Otherwise, the LM won’t be available to the
  pool’s sub-processes.

  Currently, only pools created with a ‘fork’ context can be used. If a ‘spawn’ pool is passed, it will
  be ignored and sequential decoding will be used instead.
* **num\_processes** (`int`, *optional*) —
  If `pool` is not set, number of processes on which the function should be parallelized over. Defaults
  to the number of available CPUs.
* **beam\_width** (`int`, *optional*) —
  Maximum number of beams at each step in decoding. Defaults to pyctcdecode’s DEFAULT\_BEAM\_WIDTH.
* **beam\_prune\_logp** (`int`, *optional*) —
  Beams that are much worse than best beam will be pruned Defaults to pyctcdecode’s DEFAULT\_PRUNE\_LOGP.
* **token\_min\_logp** (`int`, *optional*) —
  Tokens below this logp are skipped unless they are argmax of frame Defaults to pyctcdecode’s
  DEFAULT\_MIN\_TOKEN\_LOGP.
* **hotwords** (`list[str]`, *optional*) —
  List of words with extra importance, can be OOV for LM
* **hotword\_weight** (`int`, *optional*) —
  Weight factor for hotword importance Defaults to pyctcdecode’s DEFAULT\_HOTWORD\_WEIGHT.
* **alpha** (`float`, *optional*) —
  Weight for language model during shallow fusion
* **beta** (`float`, *optional*) —
  Weight for length score adjustment of during scoring
* **unk\_score\_offset** (`float`, *optional*) —
  Amount of log score offset for unknown tokens
* **lm\_score\_boundary** (`bool`, *optional*) —
  Whether to have kenlm respect boundaries when scoring
* **output\_word\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
  and model downsampling rate to compute the time-stamps of transcribed words.
* **n\_best** (`int`, *optional*, defaults to `1`) —
  Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list
  of lists of strings, `logit_score` will be a list of lists of floats, and `lm_score` will be a list of
  lists of floats, where the length of the outer list will correspond to the batch size and the length of
  the inner list will correspond to the number of returned hypotheses . The value should be >= 1.

  Please take a look at the Example of [decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.decode) to better understand how to
  make use of `output_word_offsets`. [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) works the same way with
  batched output.

Batch decode output logits to audio transcription with language model support.

This function makes use of Python’s multiprocessing. Currently, multiprocessing is available only on Unix
systems (see this [issue](https://github.com/kensho-technologies/pyctcdecode/issues/65)).

If you are decoding multiple batches, consider creating a `Pool` and passing it to `batch_decode`. Otherwise,
`batch_decode` will be very slow since it will create a fresh `Pool` for each call. See usage example below.

Example:
See [Decoding multiple audios](#decoding-multiple-audios).

#### decode

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L477)

( logits: ndarray beam\_width: typing.Optional[int] = None beam\_prune\_logp: typing.Optional[float] = None token\_min\_logp: typing.Optional[float] = None hotwords: typing.Optional[collections.abc.Iterable[str]] = None hotword\_weight: typing.Optional[float] = None alpha: typing.Optional[float] = None beta: typing.Optional[float] = None unk\_score\_offset: typing.Optional[float] = None lm\_score\_boundary: typing.Optional[bool] = None output\_word\_offsets: bool = False n\_best: int = 1  )

Parameters

* **logits** (`np.ndarray`) —
  The logits output vector of the model representing the log probabilities for each token.
* **beam\_width** (`int`, *optional*) —
  Maximum number of beams at each step in decoding. Defaults to pyctcdecode’s DEFAULT\_BEAM\_WIDTH.
* **beam\_prune\_logp** (`int`, *optional*) —
  A threshold to prune beams with log-probs less than best\_beam\_logp + beam\_prune\_logp. The value should
  be <= 0. Defaults to pyctcdecode’s DEFAULT\_PRUNE\_LOGP.
* **token\_min\_logp** (`int`, *optional*) —
  Tokens with log-probs below token\_min\_logp are skipped unless they are have the maximum log-prob for an
  utterance. Defaults to pyctcdecode’s DEFAULT\_MIN\_TOKEN\_LOGP.
* **hotwords** (`list[str]`, *optional*) —
  List of words with extra importance which can be missing from the LM’s vocabulary, e.g. [“huggingface”]
* **hotword\_weight** (`int`, *optional*) —
  Weight multiplier that boosts hotword scores. Defaults to pyctcdecode’s DEFAULT\_HOTWORD\_WEIGHT.
* **alpha** (`float`, *optional*) —
  Weight for language model during shallow fusion
* **beta** (`float`, *optional*) —
  Weight for length score adjustment of during scoring
* **unk\_score\_offset** (`float`, *optional*) —
  Amount of log score offset for unknown tokens
* **lm\_score\_boundary** (`bool`, *optional*) —
  Whether to have kenlm respect boundaries when scoring
* **output\_word\_offsets** (`bool`, *optional*, defaults to `False`) —
  Whether or not to output word offsets. Word offsets can be used in combination with the sampling rate
  and model downsampling rate to compute the time-stamps of transcribed words.
* **n\_best** (`int`, *optional*, defaults to `1`) —
  Number of best hypotheses to return. If `n_best` is greater than 1, the returned `text` will be a list
  of strings, `logit_score` will be a list of floats, and `lm_score` will be a list of floats, where the
  length of these lists will correspond to the number of returned hypotheses. The value should be >= 1.

  Please take a look at the example below to better understand how to make use of `output_word_offsets`.

Decode output logits to audio transcription with language model support.

Example:


```
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

### Decoding multiple audios

If you are planning to decode multiple batches of audios, you should consider using [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) and passing an instantiated `multiprocessing.Pool`.
Otherwise, [batch\_decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) performance will be slower than calling [decode()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.decode) for each audio individually, as it internally instantiates a new `Pool` for every call. See the example below:


```
>>> # Let's see how to use a user-managed pool for batch decoding multiple audios
>>> from multiprocessing import get_context
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC, infer_device
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> device = infer_device()
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
...     device = infer_device()
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

## Wav2Vec2 specific outputs

### class transformers.models.wav2vec2\_with\_lm.processing\_wav2vec2\_with\_lm.Wav2Vec2DecoderWithLMOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L47)

( text: typing.Union[list[list[str]], list[str], str] logit\_score: typing.Union[list[list[float]], list[float], float] = None lm\_score: typing.Union[list[list[float]], list[float], float] = None word\_offsets: typing.Union[list[list[list[dict[str, typing.Union[int, str]]]]], list[list[dict[str, typing.Union[int, str]]]], list[dict[str, typing.Union[int, str]]]] = None  )

Parameters

* **text** (list of `str` or `str`) —
  Decoded logits in text from. Usually the speech transcription.
* **logit\_score** (list of `float` or `float`) —
  Total logit score of the beams associated with produced text.
* **lm\_score** (list of `float`) —
  Fused lm\_score of the beams associated with produced text.
* **word\_offsets** (list of `list[dict[str, Union[int, str]]]` or `list[dict[str, Union[int, str]]]`) —
  Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
  can be used to compute time stamps for each word.

Output type of `Wav2Vec2DecoderWithLM`, with transcription.

### class transformers.modeling\_outputs.Wav2Vec2BaseModelOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_outputs.py#L1345)

( last\_hidden\_state: typing.Optional[torch.FloatTensor] = None extract\_features: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor, ...]] = None attentions: typing.Optional[tuple[torch.FloatTensor, ...]] = None  )

Parameters

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) —
  Sequence of hidden-states at the output of the last layer of the model.
* **extract\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) —
  Sequence of extracted feature vectors of the last convolutional layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Base class for models that have been trained with the Wav2Vec2 loss objective.

### class transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L84)

( loss: typing.Optional[torch.FloatTensor] = None projected\_states: typing.Optional[torch.FloatTensor] = None projected\_quantized\_states: typing.Optional[torch.FloatTensor] = None codevector\_perplexity: typing.Optional[torch.FloatTensor] = None hidden\_states: typing.Optional[tuple[torch.FloatTensor]] = None attentions: typing.Optional[tuple[torch.FloatTensor]] = None contrastive\_loss: typing.Optional[torch.FloatTensor] = None diversity\_loss: typing.Optional[torch.FloatTensor] = None  )

Parameters

* **loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) —
  Total loss as the sum of the contrastive loss (L\_m) and the diversity loss (L\_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
* **projected\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) —
  Hidden-states of the model projected to *config.proj\_codevector\_dim* that can be used to predict the masked
  projected quantized states.
* **projected\_quantized\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) —
  Quantized extracted feature vectors projected to *config.proj\_codevector\_dim* representing the positive
  target vectors for contrastive loss.
* **codevector\_perplexity** (`torch.FloatTensor` of shape `(1,)`) —
  The perplexity of the codevector distribution, used to measure the diversity of the codebook.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) —
  Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) —
  Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **contrastive\_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) —
  The contrastive loss (L\_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
* **diversity\_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) —
  The diversity loss (L\_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).

Output type of [Wav2Vec2ForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining), with potential hidden states and attentions.

## Wav2Vec2Model

### class transformers.Wav2Vec2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1346)

( config: Wav2Vec2Config  )

Parameters

* **config** ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Wav2Vec2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1432)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None mask\_time\_indices: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mask\_time\_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
  masked extracted features in *config.proj\_codevector\_dim* space.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **extract\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) — Sequence of extracted feature vectors of the last convolutional layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2Model](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Wav2Vec2ForCTC

### class transformers.Wav2Vec2ForCTC

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1767)

( config target\_lang: typing.Optional[str] = None  )

Parameters

* **config** ([Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **target\_lang** (`str`, *optional*) —
  Language id of adapter weights. Adapter weights are stored in the format adapter..safetensors or
  adapter..bin. Only relevant when using an instance of [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) with adapters. Uses ‘eng’ by
  default.

Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1845)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
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
* **labels** (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*) —
  Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
  the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
  All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`.

Returns

[transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2ForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

#### load\_adapter

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1150)

( target\_lang: str force\_load = True \*\*kwargs  )

Parameters

* **target\_lang** (`str`) —
  Has to be a language id of an existing adapter weight. Adapter weights are stored in the format
  adapter..safetensors or adapter..bin
* **force\_load** (`bool`, defaults to `True`) —
  Whether the weights shall be loaded even if `target_lang` matches `self.target_lang`.
* **cache\_dir** (`Union[str, os.PathLike]`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **local\_files\_only(`bool`,** *optional*, defaults to `False`) —
  Whether or not to only look at local files (i.e., do not try to download the model).
* **token** (`str` or `bool`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.
* **mirror** (`str`, *optional*) —
  Mirror source to accelerate downloads in China. If you are from China and have an accessibility
  problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
  Please refer to the mirror site for more information.

Load a language adapter model from a pre-trained adapter model.

Activate the special [“offline-mode”](https://huggingface.co/transformers/installation.html#offline-mode) to
use this method in a firewalled environment.

Examples:


```
>>> from transformers import Wav2Vec2ForCTC, AutoProcessor

>>> ckpt = "facebook/mms-1b-all"
>>> processor = AutoProcessor.from_pretrained(ckpt)
>>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
>>> # set specific language
>>> processor.tokenizer.set_target_lang("spa")
>>> model.load_adapter("spa")
```

## Wav2Vec2ForSequenceClassification

### class transformers.Wav2Vec2ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1923)

( config  )

Parameters

* **config** ([Wav2Vec2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1968)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
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
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
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


```
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

## Wav2Vec2ForAudioFrameClassification

### class transformers.Wav2Vec2ForAudioFrameClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2039)

( config  )

Parameters

* **config** ([Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2083)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

## Wav2Vec2ForXVector

### class transformers.Wav2Vec2ForXVector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2204)

( config  )

Parameters

* **config** ([Wav2Vec2ForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2266)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
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
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
  `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

[transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Classification hidden states before AMSoftmax.
* **embeddings** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Utterance embeddings used for vector similarity-based retrieval.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2ForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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
>>> if similarity < threshold:
...     print("Speakers are not the same!")
>>> round(similarity.item(), 2)
...
```

## Wav2Vec2ForPreTraining

### class transformers.Wav2Vec2ForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1496)

( config: Wav2Vec2Config  )

Parameters

* **config** ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a quantizer and `VQ` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1556)

( input\_values: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None mask\_time\_indices: typing.Optional[torch.BoolTensor] = None sampled\_negative\_indices: typing.Optional[torch.BoolTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_values** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
  into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
  (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
  To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion
  into a tensor of type `torch.FloatTensor`. See [Wav2Vec2Processor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **mask\_time\_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
  masked extracted features in *config.proj\_codevector\_dim* space.
* **sampled\_negative\_indices** (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*) —
  Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
  Required input for pre-training.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)`

A [transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Wav2Vec2Config](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) and inputs.

* **loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the contrastive loss (L\_m) and the diversity loss (L\_d) as stated in the [official
  paper](https://huggingface.co/papers/2006.11477).
* **projected\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) — Hidden-states of the model projected to *config.proj\_codevector\_dim* that can be used to predict the masked
  projected quantized states.
* **projected\_quantized\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`) — Quantized extracted feature vectors projected to *config.proj\_codevector\_dim* representing the positive
  target vectors for contrastive loss.
* **codevector\_perplexity** (`torch.FloatTensor` of shape `(1,)`) — The perplexity of the codevector distribution, used to measure the diversity of the codebook.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **contrastive\_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) — The contrastive loss (L\_m) as stated in the [official paper](https://huggingface.co/papers/2006.11477).
* **diversity\_loss** (`*optional*`, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`) — The diversity loss (L\_d) as stated in the [official paper](https://huggingface.co/papers/2006.11477).

The [Wav2Vec2ForPreTraining](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/wav2vec2.md)
