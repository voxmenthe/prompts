*This model was released on 2025-05-20 and added to Hugging Face Transformers on 2025-06-26.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

# Gemma3n

## Overview

[Gemma3n](https://developers.googleblog.com/en/introducing-gemma-3n/) is a multimodal model with pretrained and instruction-tuned variants, available in E4B and E2B sizes. While
large portions of the language model architecture are shared with prior Gemma releases, there are many new additions in
this model, including [Alternating Updates](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html) (AltUp), [Learned Augmented Residual Layer](https://huggingface.co/papers/2411.07501) (LAuReL),
[MatFormer](https://huggingface.co/papers/2310.07707), Per-Layer Embeddings (PLE), [Activation Sparsity with Statistical Top-k](https://huggingface.co/papers/2506.06644), and KV cache sharing. The language model uses
a similar attention pattern to [Gemma 3](./gemma3) with alternating 4 local sliding window self-attention layers for
every global self-attention layer with a maximum context length of 32k tokens. Gemma 3n introduces
[MobileNet v5][mobilenetv5] as the vision encoder, using a default resolution of 768x768 pixels, and adds a newly
trained audio encoder based on the [Universal Speech Model](https://huggingface.co/papers/2303.01037) (USM) architecture.

The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning.

You can find all the original Gemma 3n checkpoints under the [Gemma 3n](https://huggingface.co/collections/google/gemma-3n) release.

Click on the Gemma 3n models in the right sidebar for more examples of how to apply Gemma to different vision, audio,
and language tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel

transformers CLI


```
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-3n-e4b",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

## Notes

* Use [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) for image-audio-and-text, image-and-text, image-and-audio, audio-and-text,
  image-only and audio-only inputs.
* Gemma 3n supports multiple images per input, but make sure the images are correctly batched before passing them to
  the processor. Each batch should be a list of one or more images.


  ```
  url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
  url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

  messages =[
      {
          "role": "system",
          "content": [
              {"type": "text", "text": "You are a helpful assistant."}
          ]
      },
      {
          "role": "user",
          "content": [
              {"type": "image", "url": url_cow},
              {"type": "image", "url": url_cat},
              {"type": "text", "text": "Which image is cuter?"},
          ]
      },
  ]
  ```
* Text passed to the processor should have a `<image_soft_token>` token wherever an image should be inserted.
* Gemma 3n accept at most one target audio clip per input, though multiple audio clips can be provided in few-shot
  prompts, for example.
* Text passed to the processor should have a `<audio_soft_token>` token wherever an audio clip should be inserted.
* The processor has its own [apply\_chat\_template()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template) method to convert chat messages to model inputs.

## Gemma3nAudioFeatureExtractor

### class transformers.Gemma3nAudioFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/feature_extraction_gemma3n.py#L110)

( feature\_size: int = 128 sampling\_rate: int = 16000 padding\_value: float = 0.0 return\_attention\_mask: bool = True frame\_length\_ms: float = 32.0 hop\_length\_ms: float = 10.0 min\_frequency: float = 125.0 max\_frequency: float = 7600.0 preemphasis: float = 0.97 preemphasis\_htk\_flavor: bool = True fft\_overdrive: bool = True dither: float = 0.0 input\_scale\_factor: float = 1.0 mel\_floor: float = 1e-05 per\_bin\_mean: typing.Optional[collections.abc.Sequence[float]] = None per\_bin\_stddev: typing.Optional[collections.abc.Sequence[float]] = None \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 128) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  Padding value used to pad the audio. Should correspond to silences.
* **return\_attention\_mask** (`bool`, *optional*, defaults to `True`) —
  Whether to return the attention mask for the generated MEL spectrograms.
* **frame\_length\_ms** (`float`, *optional*, defaults to 32.0) —
  The length of a frame in milliseconds.
* **hop\_length\_ms** (`float`, *optional*, defaults to 10.0) —
  Length of the overlapping windows for the STFT used to obtain the Mel Frequency coefficients.
* **min\_frequency** (`float`, *optional*, defaults to 125.0) —
  The minimum frequency (in Hz) for the Mel filterbank.
* **max\_frequency** (`float`, *optional*, defaults to 7600.0) —
  The maximum frequency (in Hz) for the Mel filterbank.
* **preemphasis** (`float`, *optional*, defaults to 0.97) —
  The preemphasis coefficient.
* **preemphasis\_htk\_flavor** (`bool`, *optional*, defaults to `True`) —
  Whether to use HTK-style preemphasis.
* **fft\_overdrive** (`bool`, *optional*, defaults to `True`) —
  Whether to use FFT overdrive.
* **dither** (`float`, *optional*, defaults to 0.0) —
  Adds dithering. In other words, adds a small Gaussian noise to each frame.
  E.g. use 0.0001 to add dithering with a normal distribution centered
  around 0.0 with standard deviation 0.0001 (assuming [-1,+1] range of raw\_speech).
  The value 0.0 means no dithering.
  Dithering has similar effect as `spectrogram(mel_floor=...)`. It reduces
  the high log\_mel\_fbank values for signals with hard-zero sections,
  when VAD cutoff is present in the signal.
* **input\_scale\_factor** (`float`, *optional*, defaults to 1.0) —
  Scaling factor applied to the input waveform.
* **mel\_floor** (`float`, *optional*, defaults to 1e-05) —
  Minimum value for Mel spectrograms to avoid log(0).
* **per\_bin\_mean** (`Optional[Sequence[float]]`, *optional*) —
  Mean values for per-bin normalization.
* **per\_bin\_stddev** (`Optional[Sequence[float]]`, *optional*) —
  Standard deviation values for per-bin normalization.

An audio feature extractor Universal Speech Models <https://huggingface.co/papers/2303.01037>.

## Gemma3nProcessor

### class transformers.Gemma3nProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/processing_gemma3n.py#L40)

( feature\_extractor image\_processor tokenizer chat\_template = None audio\_seq\_length: int = 188 image\_seq\_length: int = 256 \*\*kwargs  )

Parameters

* **feature\_extractor** (`Gemma3nAudioFeatureExtractor`) —
  Feature extractor that converts raw audio waveforms into MEL spectrograms for the audio encoder. This
  should return a `BatchFeature` with `input_features` and `input_features_mask` features.
* **image\_processor** (`SiglipImageProcessorFast`) —
  Image processor that prepares batches of images for the vision encoder. This should return a `BatchFeature`
  with a `pixel_values` feature.
* **tokenizer** (`GemmaTokenizerFast`) —
  The text tokenizer for the model.
* **chat\_template** (`string`, *optional*) —
  A Jinja template for generating text prompts from a set of messages.
* **audio\_seq\_length** (int, *optional*, defaults to 188) —
  The number of audio soft tokens that will be added to the text prompt
* **image\_seq\_length** (int, *optional*, defaults to 256) —
  The number of image soft tokens that should be added to

A processor for Gemma 3n, wrapping the full capabilities of a feature extractor, image processor, and tokenizer
into a single processor.

## Gemma3nTextConfig

### class transformers.Gemma3nTextConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/configuration_gemma3n.py#L37)

( vocab\_size: int = 262400 vocab\_size\_per\_layer\_input: int = 262144 hidden\_size: int = 2048 hidden\_size\_per\_layer\_input: int = 256 intermediate\_size: typing.Union[int, collections.abc.Sequence[int]] = 16384 num\_hidden\_layers: int = 35 num\_attention\_heads: int = 8 num\_key\_value\_heads: int = 2 head\_dim: int = 256 hidden\_activation: str = 'gelu\_pytorch\_tanh' max\_position\_embeddings: int = 32768 initializer\_range: float = 0.02 rms\_norm\_eps: float = 1e-06 use\_cache: bool = True pad\_token\_id: int = 0 eos\_token\_id: int = 1 bos\_token\_id: int = 2 rope\_theta: float = 1000000.0 rope\_scaling: typing.Optional[dict[str, typing.Any]] = None rope\_local\_base\_freq: float = 10000.0 attention\_bias: bool = False attention\_dropout: float = 0.0 sliding\_window: int = 512 layer\_types: typing.Optional[collections.abc.Sequence[str]] = None final\_logit\_softcapping: float = 30.0 altup\_active\_idx: int = 0 altup\_coef\_clip: float = 120.0 altup\_correct\_scale: bool = True altup\_num\_inputs: int = 4 num\_kv\_shared\_layers: int = 15 laurel\_rank: int = 64 activation\_sparsity\_pattern: typing.Union[float, collections.abc.Sequence[float], NoneType] = None \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 262400) —
  Vocabulary size of the Gemma3nText model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [Gemma3nTextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextModel)
* **vocab\_size\_per\_layer\_input** (`int`, *optional*, defaults to 262144) —
  Vocabulary size of the per-layer text embeddings that augment the standard embeddings.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the hidden representations.
* **hidden\_size\_per\_layer\_input** (`int`, *optional*, defaults to 256) —
  Dimension of the hidden representations for per-layer emebeddings.
* **intermediate\_size** (`int` or `Sequence[int]`, *optional*, defaults to 16384) —
  Dimension of the MLP representations. MatFormer configurations may wish to provide a sequence of integers
  to account for vairable intermediate\_size values across layers. In such cases,
  `len(intermediate_size) == num_hidden_layers`.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 35) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 2) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
  converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
  by meanpooling all the original heads within that group. For more details checkout this
  [paper](https://huggingface.co/papers/2305.13245). If not specified, will default to `num_attention_heads`.
* **head\_dim** (`int`, *optional*, defaults to 256) —
  The attention head dimension.
* **hidden\_activation** (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`) —
  The non-linear activation function (function or string) in the decoder. Will default to
  `"gelu_pytorch_tanh"` if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
  activation function.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 32768) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  Padding token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 1) —
  End of stream token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 2) —
  Beginning of stream token id.
* **rope\_theta** (`float`, *optional*, defaults to 1000000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings used in gloabl attention.
  NOTE: if you apply new rope type and you expect the model to work on longer `max_position_embeddings`, we
  recommend you to update this value accordingly.
  Expected contents:
  `rope_type` (`str`):
  The sub-variant of RoPE to use. Can be one of [‘default’, ‘linear’, ‘dynamic’, ‘yarn’, ‘longrope’,
  ‘llama3’], with ‘default’ being the original RoPE implementation.
  `factor` (`float`, *optional*):
  Used with all rope types except ‘default’. The scaling factor to apply to the RoPE embeddings. In
  most scaling types, a `factor` of x will enable the model to handle sequences of length x *original maximum pre-trained length.
  `original_max_position_embeddings` (`int`,* optional*):
  Used with ‘dynamic’, ‘longrope’ and ‘llama3’. The original max position embeddings used during
  pretraining.
  `attention_factor` (`float`,* optional*):
  Used with ‘yarn’ and ‘longrope’. The scaling factor to be applied on the attention
  computation. If unspecified, it defaults to value recommended by the implementation, using the
  `factor` field to infer the suggested value.
  `beta_fast` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for extrapolation (only) in the linear
  ramp function. If unspecified, it defaults to 32.
  `beta_slow` (`float`,* optional*):
  Only used with ‘yarn’. Parameter to set the boundary for interpolation (only) in the linear
  ramp function. If unspecified, it defaults to 1.
  `short_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to short contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `long_factor` (`List[float]`,* optional*):
  Only used with ‘longrope’. The scaling factor to be applied to long contexts (<
  `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
  size divided by the number of attention heads divided by 2
  `low_freq_factor` (`float`,* optional*):
  Only used with ‘llama3’. Scaling factor applied to low frequency components of the RoPE
  `high_freq_factor` (`float`,* optional\*):
  Only used with ‘llama3’. Scaling factor applied to high frequency components of the RoPE
* **rope\_local\_base\_freq** (float, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings for local attention.
* **attention\_bias** (`bool`, defaults to `False`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.
* **sliding\_window** (`int`, *optional*, defaults to 512) —
  This is the size of the sliding window used by local attention layers.
* **layer\_types** (`Optional`, *optional*) —
  A sequence of strings defining the attention type for that layer as either “sliding\_attention” or
  “full\_attention”. If not provided, `layer_types` will de inferred from `num_hidden_layers` using a pattern
  of four “sliding\_attention” layers followed one “full\_attention”. The last layer in the model should always
  be a “full\_attention” layer.
* **final\_logit\_softcapping** (`float`, *optional*, defaults to 30.0) —
  Scaling factor when applying tanh softcapping on the logits.
* **altup\_active\_idx** (`int`, *optional*, defaults to 0) —
  The index of the prediction from which AltUp will compute additional predictions or correct
* **altup\_coef\_clip** (`float`, *optional*, defaults to 120.0) —
  The maximum amplitude of an AltUp prediction or correction coeficient weight.
* **altup\_correct\_scale** (`bool`, *optional*, defaults to `True`) —
  If True, apply the `AltUp.correct_output_scale` to the corrected prediction at `altup_active_idx`.
* **altup\_num\_inputs** (`int`, *optional*, defaults to 4) —
  The number of predictions that AltUp should be make given the input sequence.
* **num\_kv\_shared\_layers** (`int`, *optional*, defaults to 15) —
  The number of layer that share KV cache values. During the forward pass, the last `num_kv_shared_layers`
  layers in the model “share” the KV values in that each local and global layer in this range uses the KV
  cache values computed for the last local or global layer, respectively, before entering this range. The
  value should be a multiple of the attention pattern size (see `layer_types` parameter).
* **laurel\_rank** (int, *optional*, defaults to 64) —
  The intermediate size for the linear projections in the Learned Augmented Residual Layer.
* **activation\_sparsity\_pattern** (Sequence[float], *optional*) —
  The sparsity factor used to extract the top-k activations for a given layer. The provided Sequence must
  explicitly provide a sparsity value for each layer in the model. By default, the first 10 layers are
  sparse with a sparsity factor of 0.95 and the rest are dense.

This is the configuration class to store the configuration of a [Gemma3nTextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextModel). It is used to instantiate an
Gemma3nTextModel model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B, e.g.
[google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

Configuration objects that inherit from [Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig) and can be used to control the model outputs. Read
the documentation from [Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig) for more information.


```
>>> from transformers import Gemma3nTextModel, Gemma3nTextConfig

>>> # Initializing a Gemma3nText gemma3n_text-E4B style configuration
>>> configuration = Gemma3nTextConfig()

>>> # Initializing a model from the gemma3n_text-E4B style configuration
>>> model = Gemma3nTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Gemma3nVisionConfig

### class transformers.Gemma3nVisionConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/configuration_gemma3n.py#L445)

( initializer\_range: float = 0.02 do\_pooling: bool = False architecture: str = 'mobilenetv5\_300m\_enc' hidden\_size: int = 2048 vocab\_size: int = 128 vocab\_offset: int = 262144 rms\_norm\_eps: float = 1e-06 model\_args: typing.Optional[dict] = None \*\*kwargs  )

Parameters

* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **do\_pooling** (`bool`, *optional*, defaults to `False`) —
  Whether to do pooling for the last\_hidden\_state in `TimmWrapper` or not.
* **architecture** (`str`, *optional*, defaults to `"mobilenetv5_300m_enc"`) —
  Determines vision architecture for TimmWrapper.
* **hidden\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the hidden representations.
* **vocab\_size** (`int`, *optional*, defaults to 128) —
  Vocabulary size of the additional hard-token embeddings for vision model.
* **vocab\_offset** (`int`, *optional*, defaults to 262144) —
  Offset between the tokenizer vocab index for the token ids embedded by `Gemma3nMultimodalEmbedder` and the
  0-indexed `Gemma3nMultimodalEmbedder.embedding` table.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.

This is the configuration class to store the configuration for a timm backbone `TimmWrapper`. It is used to
instantiate an timm model model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B
vision tower, e.g. [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

Configuration objects inherit from [Gemma3nVisionConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nVisionConfig) and can be used to control the model outputs. Read the
documentation from [Gemma3nVisionConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nVisionConfig) for more information.

Config loads imagenet label descriptions and stores them in `id2label` attribute, `label2id` attribute for default
imagenet models is set to `None` due to occlusions in the label descriptions.

Example:


```
>>> from transformers import Gemma3nVisionConfig, TimmWrapper

>>> # Initializing a TimmWrapper gemma3n_vision-E4B-style configuration
>>> configuration = Gemma3nVisionConfig()

>>> # Initializing a gemma3n_vision-E4B-style TimmWrapper from the configuration
>>> model = TimmWrapper(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Gemma3nAudioConfig

### class transformers.Gemma3nAudioConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/configuration_gemma3n.py#L306)

( vocab\_size: int = 128 vocab\_offset: int = 262272 input\_feat\_size: int = 128 hidden\_size: int = 1536 rms\_norm\_eps: float = 1e-06 gradient\_clipping: float = 10000000000.0 conf\_attention\_chunk\_size: int = 12 conf\_attention\_context\_left: int = 13 conf\_attention\_context\_right: int = 0 conf\_attention\_logit\_cap: float = 50.0 conf\_num\_attention\_heads: int = 8 conf\_num\_hidden\_layers: int = 12 conf\_conv\_kernel\_size: int = 5 conf\_reduction\_factor: int = 4 conf\_residual\_weight: float = 0.5 sscp\_conv\_channel\_size: tuple = (128, 32) sscp\_conv\_group\_norm\_eps: float = 0.001 sscp\_conv\_kernel\_size: tuple = ((3, 3), (3, 3)) sscp\_conv\_stride\_size: tuple = ((2, 2), (2, 2)) \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 128) —
  Vocabulary size of the additional hard-token embeddings for audio model. These augment the embeddings
  included in the `Gemma3nTextModel` to provide, e.g., the end of audio and audio soft token placeholder
  tokens when converting `input_ids` to embeddings in the `Gemma3nForConditionalGeneration` model.
* **vocab\_offset** (`int`, *optional*, defaults to 262272) —
  Offset between the tokenizer vocab index for the token ids embedded by `Gemma3nMultimodalEmbedder` and the
  0-indexed `Gemma3nMultimodalEmbedder.embedding` table.
* **input\_feat\_size** (`int`, *optional*, defaults to 128) —
  The number of channels in each mel-spectrogram frame.
* **hidden\_size** (`int`, *optional*, defaults to 1536) —
  Dimension of the hidden representations.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **gradient\_clipping** (`float`, *optional*, defaults to 10000000000.0) —
  Clipping value used to stablize extremely large gradient values.
* **conf\_attention\_chunk\_size** (`int`, *optional*, defaults to 12) —
  The sub-sequence size for local attention processing inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_attention\_context\_left** (`int`, *optional*, defaults to 13) —
  The left context size of the local attention inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_attention\_context\_right** (`int`, *optional*, defaults to 0) —
  The right context size of the local attention inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_attention\_logit\_cap** (`float`, *optional*, defaults to 50.0) —
  Logit cap applied during local attention inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_num\_attention\_heads** (`int`, *optional*, defaults to 8) —
  The number of attention heads in local attention inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  The number of layers that use local attention inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_conv\_kernel\_size** (`int`, *optional*, defaults to 5) —
  Convolution kernel size for the conformer block inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_reduction\_factor** (`int`, *optional*, defaults to 4) —
  Reduction factor used in the conformer block inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **conf\_residual\_weight** (`float`, *optional*, defaults to 0.5) —
  Residual connection weight inside the Conformer (“conf”) section of the
  Universal Speech Model.
* **sscp\_conv\_channel\_size** (`tuple(int, int)`, *optional*, defaults to `(128, 32)`) —
  The channel sizes for the first and second convolutional layers in the Sub-sample Convolution Projection
  (“sscp”) section of the Universal Speech Model.
* **sscp\_conv\_group\_norm\_eps** (`float`, *optional*, defaults to 0.001) —
  Epsilon used in group normalization in the subsample convolution projection in the Sub-sample Convolution
  Projection (“sscp”) section of the Universal Speech Model.
* **sscp\_conv\_kernel\_size** (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((3, 3), (3, 3))`) —
  Kernel sizes of the two convolutional layers in the subsample convolution projection in the Sub-sample
  Convolution Projection (“sscp”) section of the Universal Speech Model. The kernel sizes are specified as a
  tuple of height and width for each layer, where the height corresponds to the time dimension and the width
  corresponds to the frequency dimension.
* **sscp\_conv\_stride\_size** (`tuple(tuple(int, int), tuple(int, int))`, *optional*, defaults to `((2, 2), (2, 2))`) —
  Stride sizes of the two convolutional layers in the subsample convolution projection in the Sub-sample
  Convolution Projection (“sscp”) section of the Universal Speech Model. The stride sizes are specified as a
  tuple of height and width for each layer, where the height corresponds to the time dimension and the width
  corresponds to the frequency dimension.

This is the configuration class to store the configuration of a `Gemma3nAudioEncoder`. It is used to instantiate
an `Gemma3nAudioEncoder` model according to the specified arguments, defining the model architecture. Instantiating
a configuration with the defaults will yield a similar configuration to that of the Gemma 3n E4B, e.g.,
[google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B).

Configuration objects that inherit from [Gemma3nAudioConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioConfig) and can be used to control the model outputs. Read
the documentation from [Gemma3nAudioConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioConfig) for more information.

Example:


```
>>> from transformers import Gemma3nAudioConfig, Gemma3nAudioEncoder

>>> # Initializing a Gemma3nAudioEncoder gemma3n_audio-E4B-style configuration
>>> configuration = Gemma3nAudioConfig()

>>> # Initializing a model from the gemma3n_audio-E4B style configuration
>>> model = Gemma3nAudioEncoder(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Gemma3nConfig

### class transformers.Gemma3nConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/configuration_gemma3n.py#L563)

( text\_config: typing.Union[transformers.models.gemma3n.configuration\_gemma3n.Gemma3nTextConfig, dict[str, typing.Any], NoneType] = None vision\_config: typing.Union[transformers.models.gemma3n.configuration\_gemma3n.Gemma3nVisionConfig, dict[str, typing.Any], NoneType] = None audio\_config: typing.Union[transformers.models.gemma3n.configuration\_gemma3n.Gemma3nAudioConfig, dict[str, typing.Any], NoneType] = None audio\_soft\_tokens\_per\_image: int = 188 vision\_soft\_tokens\_per\_image: int = 256 boi\_token\_id: int = 255999 eoi\_token\_id: int = 262144 image\_token\_id: int = 262145 boa\_token\_id: int = 256000 eoa\_token\_id: int = 262272 audio\_token\_id: int = 262273 initializer\_range: float = 0.02 \*\*kwargs  )

Parameters

* **text\_config** (`Union[Gemma3nTextConfig, dict]`, *optional*) —
  The config object of the text backbone.
* **vision\_config** (`Union[AutoConfig, dict]`, *optional*) —
  Custom vision config or dict.
* **audio\_config** (`Union[AutoConfig, dict]`, *optional*) —
  Custom audio config or dict.
* **audio\_soft\_tokens\_per\_image** (`int`, *optional*, defaults to 188) —
  The number of soft tokens per audio clip.
* **vision\_soft\_tokens\_per\_image** (`int`, *optional*, defaults to 256) —
  The number of soft tokens per image.
* **boi\_token\_id** (`int`, *optional*, defaults to 255999) —
  The begin-of-image token index to wrap the image prompt.
* **eoi\_token\_id** (`int`, *optional*, defaults to 262144) —
  The end-of-image token index to wrap the image prompt.
* **image\_token\_id** (`int`, *optional*, defaults to 262145) —
  The image token index to encode the image prompt.
* **boa\_token\_id** (`int`, *optional*, defaults to 256000) —
  The begin-of-audio token index to wrap the audio prompt.
* **eoa\_token\_id** (`int`, *optional*, defaults to 262272) —
  The end-of-audio token index to wrap the audio prompt.
* **audio\_token\_id** (`int`, *optional*, defaults to 262273) —
  The audio token index to encode the audio prompt.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration). It is used to
instantiate a Gemma3nForConditionalGeneration according to the specified arguments, defining the model
architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
Gemma3n-E4B.

e.g. [google/gemma-3n-E4B](https://huggingface.co/google/gemma-3n-E4B)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Gemma3nForConditionalGeneration, Gemma3nConfig, Gemma3nTextConfig

>>> # Initializing a MobileNet vision config, which is loaded from TIMM
>>> vision_config = Gemma3nVisionConfig()

>>> # Initializing a Gemma3n Audio config
>>> audio_config = Gemma3nAudioConfig()

>>> # Initializing a Gemma3n Text config
>>> text_config = Gemma3nTextConfig()

>>> # Initializing a Gemma3n gemma-3-4b style configuration
>>> configuration = Gemma3nConfig(text_config, vision_config, audio_config)

>>> # Initializing a model from the gemma-3-4b style configuration
>>> model = Gemma3nTextConfig(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Gemma3nTextModel

### class transformers.Gemma3nTextModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L1511)

( config: Gemma3nTextConfig  )

Parameters

* **config** ([Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 3n language model without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L1571)

( input\_ids: typing.Optional[torch.LongTensor] = None per\_layer\_inputs: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **per\_layer\_inputs** (`torch.Tensor`, *optional*, defaults to None) —
  Pre-computed per-layer embeddings. If None, they are derived from input\_ids if provided.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig)) and inputs.

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

The [Gemma3nTextModel](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## Gemma3nModel

### class transformers.Gemma3nModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L1912)

( config: Gemma3nConfig  )

Parameters

* **config** ([Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 3n model comprising a vision backbone, an audio backbone, and a language model without a
language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L2011)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None input\_features\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None \*\*lm\_kwargs  )

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
(masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Gemma3nForConditionalGeneration

>>> model = Gemma3nForConditionalGeneration.from_pretrained("google/gemma3n2-3b-mix-224")
>>> processor = AutoProcessor.from_pretrained("google/gemma3n2-3b-mix-224")

>>> prompt = "Where is the cat standing?"
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs,)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Where is the cat standing?\nsnow"
```

## Gemma3nForCausalLM

### class transformers.Gemma3nForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L1759)

( config: Gemma3nTextConfig  )

Parameters

* **config** ([Gemma3nTextConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nTextConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 3n language model with a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L1776)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [Gemma3nForCausalLM](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, Gemma3nForCausalLM

>>> model = Gemma3nForCausalLM.from_pretrained("google/gemma-2-9b")
>>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

>>> prompt = "What is your favorite condiment?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is your favorite condiment?"
```

## Gemma3nForConditionalGeneration

### class transformers.Gemma3nForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L2176)

( config: Gemma3nConfig  )

Parameters

* **config** ([Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The base Gemma 3n model comprising a vision backbone, an audio backbone, a language model, and a language modeling
head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3n/modeling_gemma3n.py#L2215)

( input\_ids: typing.Optional[torch.LongTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None input\_features: typing.Optional[torch.FloatTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None input\_features\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Union[list[torch.FloatTensor], transformers.cache\_utils.Cache, NoneType] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*lm\_kwargs  ) → `transformers.models.gemma3n.modeling_gemma3n.Gemma3nCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor). See [SiglipImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Gemma3nProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nProcessor) uses
  [SiglipImageProcessor](/docs/transformers/v4.56.2/en/model_doc/siglip#transformers.SiglipImageProcessor) for processing images).
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) —
  The tensors corresponding to the input audio features. Audio features can be obtained using
  [Gemma3nAudioFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioFeatureExtractor). See `Gemma3nAudioFeatureExtractor.__call__()` for details ([Gemma3nProcessor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nProcessor) uses
  [Gemma3nAudioFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nAudioFeatureExtractor) for processing audios).
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **input\_features\_mask** (`torch.Tensor`, *optional*, defaults to None) —
  The attention mask for the input audio.
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`Union[list[torch.FloatTensor], ~cache_utils.Cache, NoneType]`) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are
  ignored (masked), the loss is only computed for the tokens with labels in
  `[0, ..., config.text_config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.gemma3n.modeling_gemma3n.Gemma3nCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.gemma3n.modeling_gemma3n.Gemma3nCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Gemma3nConfig](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder after projecting last hidden state.
* **audio\_hidden\_states** (`torch.FloatTensor`, *optional*) — A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  audio\_hidden\_states of the model produced by the audio encoder and after projecting the last hidden state.

The [Gemma3nForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/gemma3n#transformers.Gemma3nForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

>>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
>>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

>>> messages = [
...     {
...         "role": "system",
...         "content": [
...             {"type": "text", "text": "You are a helpful assistant."}
...         ]
...     },
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenizer=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gemma3n.md)
