*This model was released on 2023-11-30 and added to Hugging Face Transformers on 2024-01-18.*

# Wav2Vec2-BERT

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The [Wav2Vec2-BERT](https://huggingface.co/papers/2312.05187) model was proposed in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/) by the Seamless Communication team from Meta AI.

This model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. It requires finetuning to be used for downstream tasks such as Automatic Speech Recognition (ASR), or Audio Classification.

The official results of the model can be found in Section 3.2.1 of the paper.

The abstract from the paper is the following:

*Recent advancements in automatic speech translation have dramatically expanded language coverage, improved multimodal capabilities, and enabled a wide range of tasks and functionalities. That said, large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model—SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. The expanded version of SeamlessAlign adds 114,800 hours of automatically aligned data for a total of 76 languages. SeamlessM4T v2 provides the foundation on which our two newest models, SeamlessExpressive and SeamlessStreaming, are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one’s voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention (EMMA) mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To understand the performance of these models, we combined novel and modified versions of existing automatic metrics to evaluate prosody, latency, and robustness. For human evaluations, we adapted existing protocols tailored for measuring the most relevant attributes in the preservation of meaning, naturalness, and expressivity. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. In sum, Seamless gives us a pivotal look at the technical foundation needed to turn the Universal Speech Translator from a science fiction concept into a real-world technology. Finally, contributions in this work—including models, code, and a watermark detector—are publicly released and accessible at the link below.*

This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## Usage tips

* Wav2Vec2-BERT follows the same architecture as Wav2Vec2-Conformer, but employs a causal depthwise convolutional layer and uses as input a mel-spectrogram representation of the audio instead of the raw waveform.
* Wav2Vec2-BERT can use either no relative position embeddings, Shaw-like position embeddings, Transformer-XL-like position embeddings, or
  rotary position embeddings by setting the correct `config.position_embeddings_type`.
* Wav2Vec2-BERT also introduces a Conformer-based adapter network instead of a simple convolutional network.

## Resources

Automatic Speech Recognition

* [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition).
* You can also adapt these notebooks on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).

Audio Classification

* [Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification) can be used by adapting this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification).
* See also: [Audio classification task guide](../tasks/audio_classification)

## Wav2Vec2BertConfig

### class transformers.Wav2Vec2BertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py#L24)

( vocab\_size = None hidden\_size = 1024 num\_hidden\_layers = 24 num\_attention\_heads = 16 intermediate\_size = 4096 feature\_projection\_input\_dim = 160 hidden\_act = 'swish' hidden\_dropout = 0.0 activation\_dropout = 0.0 attention\_dropout = 0.0 feat\_proj\_dropout = 0.0 final\_dropout = 0.1 layerdrop = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 apply\_spec\_augment = True mask\_time\_prob = 0.05 mask\_time\_length = 10 mask\_time\_min\_masks = 2 mask\_feature\_prob = 0.0 mask\_feature\_length = 10 mask\_feature\_min\_masks = 0 ctc\_loss\_reduction = 'sum' ctc\_zero\_infinity = False use\_weighted\_layer\_sum = False classifier\_proj\_size = 768 tdnn\_dim = (512, 512, 512, 512, 1500) tdnn\_kernel = (5, 3, 3, 1, 1) tdnn\_dilation = (1, 2, 3, 1, 1) xvector\_output\_dim = 512 pad\_token\_id = 0 bos\_token\_id = 1 eos\_token\_id = 2 add\_adapter = False adapter\_kernel\_size = 3 adapter\_stride = 2 num\_adapter\_layers = 1 adapter\_act = 'relu' use\_intermediate\_ffn\_before\_adapter = False output\_hidden\_size = None position\_embeddings\_type = 'relative\_key' rotary\_embedding\_base = 10000 max\_source\_positions = 5000 left\_max\_position\_embeddings = 64 right\_max\_position\_embeddings = 8 conv\_depthwise\_kernel\_size = 31 conformer\_conv\_dropout = 0.1 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*) —
  Vocabulary size of the Wav2Vec2Bert model. Defines the number of different tokens that can be
  represented by the `inputs_ids` passed when calling [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel). Vocabulary size of the
  model. Defines the different tokens that can be represented by the *inputs\_ids* passed to the forward
  method of [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel).
* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the encoder layers and the pooler layer.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **feature\_projection\_input\_dim** (`int`, *optional*, defaults to 160) —
  Input dimension of this model, i.e the dimension after processing input audios with [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) or [Wav2Vec2BertProcessor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertProcessor).
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"swish"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
* **hidden\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for activations inside the fully connected layer.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **feat\_proj\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for the feature projection.
* **final\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for the final projection layer of [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC).
* **layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>) for more
  details.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **apply\_spec\_augment** (`bool`, *optional*, defaults to `True`) —
  Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
  [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
  Recognition](https://huggingface.co/papers/1904.08779).
* **mask\_time\_prob** (`float`, *optional*, defaults to 0.05) —
  Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
  procedure generates ``` mask_time_prob*len(time_axis)/mask_time_length ``independent masks over the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector span to be masked, *mask_time_prob* should be ``` prob\_vector\_start\*mask\_time\_length`. Note that overlap may decrease the actual percentage of masked vectors. This is only relevant if` apply\_spec\_augment is True`.
* **mask\_time\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the time axis.
* **mask\_time\_min\_masks** (`int`, *optional*, defaults to 2) —
  The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
  irrespectively of `mask_feature_prob`. Only relevant if `mask_time_prob*len(time_axis)/mask_time_length < mask_time_min_masks`.
* **mask\_feature\_prob** (`float`, *optional*, defaults to 0.0) —
  Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
  masking procedure generates `mask_feature_prob*len(feature_axis)/mask_time_length` independent masks over
  the axis. If reasoning from the probability of each feature vector to be chosen as the start of the vector
  span to be masked, *mask\_feature\_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
  may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
* **mask\_feature\_length** (`int`, *optional*, defaults to 10) —
  Length of vector span along the feature axis.
* **mask\_feature\_min\_masks** (`int`, *optional*, defaults to 0) —
  The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
  step, irrespectively of `mask_feature_prob`. Only relevant if
  `mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks`.
* **ctc\_loss\_reduction** (`str`, *optional*, defaults to `"sum"`) —
  Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
  instance of [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC).
* **ctc\_zero\_infinity** (`bool`, *optional*, defaults to `False`) —
  Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
  occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
  of [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC).
* **use\_weighted\_layer\_sum** (`bool`, *optional*, defaults to `False`) —
  Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
  instance of [Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification).
* **classifier\_proj\_size** (`int`, *optional*, defaults to 768) —
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
* **pad\_token\_id** (`int`, *optional*, defaults to 0) — The id of the *beginning-of-stream* token.
* **bos\_token\_id** (`int`, *optional*, defaults to 1) — The id of the *padding* token.
* **eos\_token\_id** (`int`, *optional*, defaults to 2) — The id of the *end-of-stream* token.
* **add\_adapter** (`bool`, *optional*, defaults to `False`) —
  Whether a convolutional attention network should be stacked on top of the Wav2Vec2Bert Encoder. Can be very
  useful for warm-starting Wav2Vec2Bert for SpeechEncoderDecoder models.
* **adapter\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adapter\_stride** (`int`, *optional*, defaults to 2) —
  Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **num\_adapter\_layers** (`int`, *optional*, defaults to 1) —
  Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is True`.
* **adapter\_act** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the adapter layers. If string, `"gelu"`,
  `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
* **use\_intermediate\_ffn\_before\_adapter** (`bool`, *optional*, defaults to `False`) —
  Whether an intermediate feed-forward block should be stacked on top of the Wav2Vec2Bert Encoder and before the adapter network.
  Only relevant if `add_adapter is True`.
* **output\_hidden\_size** (`int`, *optional*) —
  Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
  if `add_adapter is True`.
* **position\_embeddings\_type** (`str`, *optional*, defaults to `"relative_key"`) —
  Can be specified to :
  + `rotary`, for rotary position embeddings.
  + `relative`, for relative position embeddings.
  + `relative_key`, for relative position embeddings as defined by Shaw in [Self-Attention
    with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
    If left to `None`, no relative position embeddings is applied.
* **rotary\_embedding\_base** (`int`, *optional*, defaults to 10000) —
  If `"rotary"` position embeddings are used, defines the size of the embedding base.
* **max\_source\_positions** (`int`, *optional*, defaults to 5000) —
  if `"relative"` position embeddings are used, defines the maximum source input positions.
* **left\_max\_position\_embeddings** (`int`, *optional*, defaults to 64) —
  If `"relative_key"` (aka Shaw) position embeddings are used, defines the left clipping value for relative positions.
* **right\_max\_position\_embeddings** (`int`, *optional*, defaults to 8) —
  If `"relative_key"` (aka Shaw) position embeddings are used, defines the right clipping value for relative positions.
* **conv\_depthwise\_kernel\_size** (`int`, *optional*, defaults to 31) —
  Kernel size of convolutional depthwise 1D layer in Conformer blocks.
* **conformer\_conv\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all convolutional layers in Conformer blocks.

This is the configuration class to store the configuration of a [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel). It is used to
instantiate an Wav2Vec2Bert model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2Bert
[facebook/wav2vec2-bert-rel-pos-large](https://huggingface.co/facebook/wav2vec2-bert-rel-pos-large)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

>>> # Initializing a Wav2Vec2Bert facebook/wav2vec2-bert-rel-pos-large style configuration
>>> configuration = Wav2Vec2BertConfig()

>>> # Initializing a model (with random weights) from the facebook/wav2vec2-bert-rel-pos-large style configuration
>>> model = Wav2Vec2BertModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Wav2Vec2BertProcessor

### class transformers.Wav2Vec2BertProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py#L32)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** (`SeamlessM4TFeatureExtractor`) —
  An instance of [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor). The feature extractor is a required input.
* **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  An instance of [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). The tokenizer is a required input.

Constructs a Wav2Vec2-BERT processor which wraps a Wav2Vec2-BERT feature extractor and a Wav2Vec2 CTC tokenizer into a single
processor.

[Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) offers all the functionalities of [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
See the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py#L72)

( audio: typing.Union[ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), list['np.ndarray'], list['torch.Tensor']] = None text: typing.Union[str, list[str], NoneType] = None images = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.wav2vec2\_bert.processing\_wav2vec2\_bert.Wav2Vec2BertProcessorKwargs]  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **audio** (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
  of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
  and T the sample length of the audio.
* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_features** — Audio input features to be fed to a model. Returned when `audio` is not `None`.
* **attention\_mask** — List of indices specifying which timestamps should be attended to by the model when `audio` is not `None`.
  When only `text` is specified, returns the token attention mask.
* **labels** — List of token ids to be fed to a model. Returned when both `text` and `audio` are not `None`.
* **input\_ids** — List of token ids to be fed to a model. Returned when `text` is not `None` and `audio` is `None`.

Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
and `kwargs` arguments to SeamlessM4TFeatureExtractor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) if `audio` is not
`None` to pre-process the audio. To prepare the target sequences(s), this method forwards the `text` and `kwargs` arguments to
PreTrainedTokenizer’s [**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) if `text` is not `None`. Please refer to the docstring of the above two methods for more information.

#### pad

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py#L126)

( input\_features = None labels = None \*\*kwargs  )

If `input_features` is not `None`, this method forwards the `input_features` and `kwargs` arguments to SeamlessM4TFeatureExtractor’s [pad()](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) to pad the input features.
If `labels` is not `None`, this method forwards the `labels` and `kwargs` arguments to PreTrainedTokenizer’s [pad()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad) to pad the label(s).
Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/processing_wav2vec2_bert.py#L53)

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

## Wav2Vec2BertModel

### class transformers.Wav2Vec2BertModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L920)

( config: Wav2Vec2BertConfig  )

Parameters

* **config** ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Wav2Vec2 Bert Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L987)

( input\_features: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None mask\_time\_indices: typing.Optional[torch.FloatTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). See [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) for details ([Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) uses
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) for processing audios).
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
elements depending on the configuration ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **extract\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`) — Sequence of extracted feature vectors of the last convolutional layer of the model.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2BertModel](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Wav2Vec2BertForCTC

### class transformers.Wav2Vec2BertForCTC

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1049)

( config target\_lang: typing.Optional[str] = None  )

Parameters

* **config** ([Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **target\_lang** (`str`, *optional*) —
  Language id of adapter weights. Adapter weights are stored in the format adapter..safetensors or
  adapter..bin. Only relevant when using an instance of [UniSpeechSatForCTC](/docs/transformers/v4.56.2/en/model_doc/unispeech-sat#transformers.UniSpeechSatForCTC) with adapters. Uses ‘eng’ by
  default.

Wav2Vec2Bert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1079)

( input\_features: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). See [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) for details ([Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) uses
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) for processing audios).
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
elements depending on the configuration ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2BertForCTC](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, Wav2Vec2BertForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")
>>> model = Wav2Vec2BertForCTC.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")

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

## Wav2Vec2BertForSequenceClassification

### class transformers.Wav2Vec2BertForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1159)

( config  )

Parameters

* **config** ([Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
SUPERB Keyword Spotting.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1185)

( input\_features: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). See [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) for details ([Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) uses
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) for processing audios).
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
elements depending on the configuration ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2BertForSequenceClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example of single-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, Wav2Vec2BertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")
>>> model = Wav2Vec2BertForSequenceClassification.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = Wav2Vec2BertForSequenceClassification.from_pretrained("facebook/wav2vec2-bert-rel-pos-large", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:


```
>>> import torch
>>> from transformers import AutoTokenizer, Wav2Vec2BertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")
>>> model = Wav2Vec2BertForSequenceClassification.from_pretrained("facebook/wav2vec2-bert-rel-pos-large", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = Wav2Vec2BertForSequenceClassification.from_pretrained(
...     "facebook/wav2vec2-bert-rel-pos-large", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## Wav2Vec2BertForAudioFrameClassification

### class transformers.Wav2Vec2BertForAudioFrameClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1250)

( config  )

Parameters

* **config** ([Wav2Vec2BertForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForAudioFrameClassification)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Wav2Vec2 Bert Model with a frame classification head on top for tasks like Speaker Diarization.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1275)

( input\_features: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). See [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) for details ([Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) uses
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) for processing audios).
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
elements depending on the configuration ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2BertForAudioFrameClassification](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")
>>> model = Wav2Vec2BertForAudioFrameClassification.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")

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

## Wav2Vec2BertForXVector

### class transformers.Wav2Vec2BertForXVector

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1390)

( config  )

Parameters

* **config** ([Wav2Vec2BertForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForXVector)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2Bert Model with an XVector feature extraction head on top for tasks like Speaker Verification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L1433)

( input\_features: typing.Optional[torch.Tensor] attention\_mask: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). See [Wav2Vec2FeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) for details ([Wav2Vec2Processor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) uses
  [Wav2Vec2FeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) for processing audios).
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
elements depending on the configuration ([Wav2Vec2BertConfig](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Classification hidden states before AMSoftmax.
* **embeddings** (`torch.FloatTensor` of shape `(batch_size, config.xvector_output_dim)`) — Utterance embeddings used for vector similarity-based retrieval.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
  shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Wav2Vec2BertForXVector](/docs/transformers/v4.56.2/en/model_doc/wav2vec2-bert#transformers.Wav2Vec2BertForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2BertForXVector
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")
>>> model = Wav2Vec2BertForXVector.from_pretrained("facebook/wav2vec2-bert-rel-pos-large")

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

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/wav2vec2-bert.md)
