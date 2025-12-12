# Exporting 🤗 Transformers models to ONNX

🤗 Transformers provides a `transformers.onnx` package that enables you to
convert model checkpoints to an ONNX graph by leveraging configuration objects.

See the [guide](../serialization) on exporting 🤗 Transformers models for more
details.

## ONNX Configurations

We provide three abstract classes that you should inherit from, depending on the
type of model architecture you wish to export:

* Encoder-based models inherit from [OnnxConfig](/docs/transformers/v4.56.2/en/main_classes/onnx#transformers.onnx.OnnxConfig)
* Decoder-based models inherit from [OnnxConfigWithPast](/docs/transformers/v4.56.2/en/main_classes/onnx#transformers.onnx.OnnxConfigWithPast)
* Encoder-decoder models inherit from [OnnxSeq2SeqConfigWithPast](/docs/transformers/v4.56.2/en/main_classes/onnx#transformers.onnx.OnnxSeq2SeqConfigWithPast)

### OnnxConfig

### class transformers.onnx.OnnxConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L69)

( config: PretrainedConfig task: str = 'default' patching\_specs: typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None  )

Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.

#### flatten\_output\_collection\_property

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L427)

( name: str field: Iterable  ) → (dict[str, Any])

Parameters

* **name** — The name of the nested structure
* **field** — The structure to, potentially, be flattened

Returns

(dict[str, Any])

Outputs with flattened structure and key mapping this new structure.

Flatten any potential nested structure expanding the name of the field with the index of the element within the
structure.

#### from\_model\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L130)

( config: PretrainedConfig task: str = 'default'  )

Parameters

* **config** — The model’s configuration to use when exporting to ONNX

Instantiate a OnnxConfig for a specific model

#### generate\_dummy\_inputs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L283)

( preprocessor: typing.Union[ForwardRef('PreTrainedTokenizerBase'), ForwardRef('FeatureExtractionMixin'), ForwardRef('ImageProcessingMixin')] batch\_size: int = -1 seq\_length: int = -1 num\_choices: int = -1 is\_pair: bool = False framework: typing.Optional[transformers.utils.generic.TensorType] = None num\_channels: int = 3 image\_width: int = 40 image\_height: int = 40 sampling\_rate: int = 22050 time\_duration: float = 5.0 frequency: int = 220 tokenizer: typing.Optional[ForwardRef('PreTrainedTokenizerBase')] = None  )

Parameters

* **preprocessor** — ([PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase), [FeatureExtractionMixin](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin), or [ImageProcessingMixin](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.ImageProcessingMixin)):
  The preprocessor associated with this model configuration.
* **batch\_size** (`int`, *optional*, defaults to -1) —
  The batch size to export the model for (-1 means dynamic axis).
* **num\_choices** (`int`, *optional*, defaults to -1) —
  The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
* **seq\_length** (`int`, *optional*, defaults to -1) —
  The sequence length to export the model for (-1 means dynamic axis).
* **is\_pair** (`bool`, *optional*, defaults to `False`) —
  Indicate if the input is a pair (sentence 1, sentence 2)
* **framework** (`TensorType`, *optional*, defaults to `None`) —
  The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
* **num\_channels** (`int`, *optional*, defaults to 3) —
  The number of channels of the generated images.
* **image\_width** (`int`, *optional*, defaults to 40) —
  The width of the generated images.
* **image\_height** (`int`, *optional*, defaults to 40) —
  The height of the generated images.
* **sampling\_rate** (`int`, *optional* defaults to 22050) —
  The sampling rate for audio data generation.
* **time\_duration** (`float`, *optional* defaults to 5.0) —
  Total seconds of sampling for audio data generation.
* **frequency** (`int`, *optional* defaults to 220) —
  The desired natural frequency of generated audio.

Generate inputs to provide to the ONNX exporter for the specific framework

#### generate\_dummy\_inputs\_onnxruntime

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L403)

( reference\_model\_inputs: Mapping  ) → `Mapping[str, Tensor]`

Parameters

* **reference\_model\_inputs** ([`Mapping[str, Tensor]`) —
  Reference inputs for the model.

Returns

`Mapping[str, Tensor]`

The mapping holding the kwargs to provide to the model’s forward function

Generate inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq
models which have the encoder and decoder exported as separate ONNX files.

#### use\_external\_data\_format

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L244)

( num\_parameters: int  )

Parameters

* **num\_parameters** — Number of parameter on the model

Flag indicating if the model requires using external data format

### OnnxConfigWithPast

### class transformers.onnx.OnnxConfigWithPast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L446)

( config: PretrainedConfig task: str = 'default' patching\_specs: typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None use\_past: bool = False  )

#### fill\_with\_past\_key\_values\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L553)

( inputs\_or\_outputs: Mapping direction: str inverted\_values\_shape: bool = False  )

Parameters

* **inputs\_or\_outputs** — The mapping to fill.
* **direction** — either “inputs” or “outputs”, it specifies whether input\_or\_outputs is the input mapping or the
  output mapping, this is important for axes naming.
* **inverted\_values\_shape** —
  If `True`, store values on dynamic axis 1, else on axis 2.

Fill the input\_or\_outputs mapping with past\_key\_values dynamic axes considering.

#### with\_past

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L457)

( config: PretrainedConfig task: str = 'default'  )

Parameters

* **config** — The underlying model’s config to use when exporting to ONNX

Instantiate a OnnxConfig with `use_past` attribute set to True

### OnnxSeq2SeqConfigWithPast

### class transformers.onnx.OnnxSeq2SeqConfigWithPast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/config.py#L593)

( config: PretrainedConfig task: str = 'default' patching\_specs: typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None use\_past: bool = False  )

## ONNX Features

Each ONNX configuration is associated with a set of *features* that enable you
to export models for different types of topologies or tasks.

### FeaturesManager

### class transformers.onnx.FeaturesManager

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L85)

( )

#### check\_supported\_model\_or\_raise

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L711)

( model: typing.Union[ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel')] feature: str = 'default'  )

Parameters

* **model** — The model to export.
* **feature** — The name of the feature to check if it is available.

Check whether or not the model has the requested features.

#### determine\_framework

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L628)

( model: str framework: typing.Optional[str] = None  )

Parameters

* **model** (`str`) —
  The name of the model to export.
* **framework** (`str`, *optional*, defaults to `None`) —
  The framework to use for the export. See above for priority if none provided.

Determines the framework to use for the export.

The priority is in the following order:

1. User input via `framework`.
2. If local checkpoint is provided, use the same framework as the checkpoint.
3. Available framework in environment, with priority given to PyTorch

#### get\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L736)

( model\_type: str feature: str  ) → `OnnxConfig`

Parameters

* **model\_type** (`str`) —
  The model type to retrieve the config for.
* **feature** (`str`) —
  The feature to retrieve the config for.

Returns

`OnnxConfig`

config for the combination

Gets the OnnxConfig for a model\_type and feature combination.

#### get\_model\_class\_for\_feature

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L601)

( feature: str framework: str = 'pt'  )

Parameters

* **feature** (`str`) —
  The feature required.
* **framework** (`str`, *optional*, defaults to `"pt"`) —
  The framework to use for the export.

Attempts to retrieve an AutoModel class from a feature name.

#### get\_model\_from\_feature

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L678)

( feature: str model: str framework: typing.Optional[str] = None cache\_dir: typing.Optional[str] = None  )

Parameters

* **feature** (`str`) —
  The feature required.
* **model** (`str`) —
  The name of the model to export.
* **framework** (`str`, *optional*, defaults to `None`) —
  The framework to use for the export. See `FeaturesManager.determine_framework` for the priority should
  none be provided.

Attempts to retrieve a model from a model’s name and the feature to be enabled.

#### get\_supported\_features\_for\_model\_type

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/onnx/features.py#L556)

( model\_type: str model\_name: typing.Optional[str] = None  )

Parameters

* **model\_type** (`str`) —
  The model type to retrieve the supported features for.
* **model\_name** (`str`, *optional*) —
  The name attribute of the model object, only used for the exception message.

Tries to retrieve the feature -> OnnxConfig constructor map from the model type.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/onnx.md)
