# Exporting 🤗 Transformers models to ONNX

🤗 Transformers provides a `transformers.onnx` package that enables you to
convert model checkpoints to an ONNX graph by leveraging configuration objects.

See the [guide](../serialization) on exporting 🤗 Transformers models for more
details.

## ONNX Configurations

We provide three abstract classes that you should inherit from, depending on the
type of model architecture you wish to export:

* Encoder-based models inherit from [OnnxConfig](/docs/transformers/main/en/main_classes/onnx#transformers.onnx.OnnxConfig)
* Decoder-based models inherit from [OnnxConfigWithPast](/docs/transformers/main/en/main_classes/onnx#transformers.onnx.OnnxConfigWithPast)
* Encoder-decoder models inherit from [OnnxSeq2SeqConfigWithPast](/docs/transformers/main/en/main_classes/onnx#transformers.onnx.OnnxSeq2SeqConfigWithPast)

### OnnxConfig[[transformers.onnx.OnnxConfig]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.onnx.OnnxConfig</name><anchor>transformers.onnx.OnnxConfig</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L69</source><parameters>[{"name": "config", "val": ": PretrainedConfig"}, {"name": "task", "val": ": str = 'default'"}, {"name": "patching_specs", "val": ": typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None"}]</parameters></docstring>

Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>flatten_output_collection_property</name><anchor>transformers.onnx.OnnxConfig.flatten_output_collection_property</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L424</source><parameters>[{"name": "name", "val": ": str"}, {"name": "field", "val": ": Iterable"}]</parameters><paramsdesc>- **name** -- The name of the nested structure
- **field** -- The structure to, potentially, be flattened</paramsdesc><paramgroups>0</paramgroups><rettype>(dict[str, Any])</rettype><retdesc>Outputs with flattened structure and key mapping this new structure.</retdesc></docstring>

Flatten any potential nested structure expanding the name of the field with the index of the element within the
structure.








</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>from_model_config</name><anchor>transformers.onnx.OnnxConfig.from_model_config</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L130</source><parameters>[{"name": "config", "val": ": PretrainedConfig"}, {"name": "task", "val": ": str = 'default'"}]</parameters><paramsdesc>- **config** -- The model's configuration to use when exporting to ONNX</paramsdesc><paramgroups>0</paramgroups><retdesc>OnnxConfig for this model</retdesc></docstring>

Instantiate a OnnxConfig for a specific model






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>generate_dummy_inputs</name><anchor>transformers.onnx.OnnxConfig.generate_dummy_inputs</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L283</source><parameters>[{"name": "preprocessor", "val": ": typing.Union[ForwardRef('PreTrainedTokenizerBase'), ForwardRef('FeatureExtractionMixin'), ForwardRef('ImageProcessingMixin')]"}, {"name": "batch_size", "val": ": int = -1"}, {"name": "seq_length", "val": ": int = -1"}, {"name": "num_choices", "val": ": int = -1"}, {"name": "is_pair", "val": ": bool = False"}, {"name": "num_channels", "val": ": int = 3"}, {"name": "image_width", "val": ": int = 40"}, {"name": "image_height", "val": ": int = 40"}, {"name": "sampling_rate", "val": ": int = 22050"}, {"name": "time_duration", "val": ": float = 5.0"}, {"name": "frequency", "val": ": int = 220"}, {"name": "tokenizer", "val": ": typing.Optional[ForwardRef('PreTrainedTokenizerBase')] = None"}]</parameters><paramsdesc>- **preprocessor** -- ([PreTrainedTokenizerBase](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase), [FeatureExtractionMixin](/docs/transformers/main/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin), or [ImageProcessingMixin](/docs/transformers/main/en/main_classes/image_processor#transformers.ImageProcessingMixin)):
  The preprocessor associated with this model configuration.
- **batch_size** (`int`, *optional*, defaults to -1) --
  The batch size to export the model for (-1 means dynamic axis).
- **num_choices** (`int`, *optional*, defaults to -1) --
  The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
- **seq_length** (`int`, *optional*, defaults to -1) --
  The sequence length to export the model for (-1 means dynamic axis).
- **is_pair** (`bool`, *optional*, defaults to `False`) --
  Indicate if the input is a pair (sentence 1, sentence 2)
- **num_channels** (`int`, *optional*, defaults to 3) --
  The number of channels of the generated images.
- **image_width** (`int`, *optional*, defaults to 40) --
  The width of the generated images.
- **image_height** (`int`, *optional*, defaults to 40) --
  The height of the generated images.
- **sampling_rate** (`int`, *optional* defaults to 22050) --
  The sampling rate for audio data generation.
- **time_duration** (`float`, *optional* defaults to 5.0) --
  Total seconds of sampling for audio data generation.
- **frequency** (`int`, *optional* defaults to 220) --
  The desired natural frequency of generated audio.</paramsdesc><paramgroups>0</paramgroups><retdesc>Mapping[str, Tensor] holding the kwargs to provide to the model's forward function</retdesc></docstring>

Generate inputs to provide to the ONNX exporter






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>generate_dummy_inputs_onnxruntime</name><anchor>transformers.onnx.OnnxConfig.generate_dummy_inputs_onnxruntime</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L400</source><parameters>[{"name": "reference_model_inputs", "val": ": Mapping"}]</parameters><paramsdesc>- **reference_model_inputs** ([`Mapping[str, Tensor]`) --
  Reference inputs for the model.</paramsdesc><paramgroups>0</paramgroups><rettype>`Mapping[str, Tensor]`</rettype><retdesc>The mapping holding the kwargs to provide to the model's forward function</retdesc></docstring>

Generate inputs for ONNX Runtime using the reference model inputs. Override this to run inference with seq2seq
models which have the encoder and decoder exported as separate ONNX files.








</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>use_external_data_format</name><anchor>transformers.onnx.OnnxConfig.use_external_data_format</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L244</source><parameters>[{"name": "num_parameters", "val": ": int"}]</parameters><paramsdesc>- **num_parameters** -- Number of parameter on the model</paramsdesc><paramgroups>0</paramgroups><retdesc>True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise</retdesc></docstring>

Flag indicating if the model requires using external data format






</div></div>

### OnnxConfigWithPast[[transformers.onnx.OnnxConfigWithPast]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.onnx.OnnxConfigWithPast</name><anchor>transformers.onnx.OnnxConfigWithPast</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L443</source><parameters>[{"name": "config", "val": ": PretrainedConfig"}, {"name": "task", "val": ": str = 'default'"}, {"name": "patching_specs", "val": ": typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None"}, {"name": "use_past", "val": ": bool = False"}]</parameters></docstring>



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>fill_with_past_key_values_</name><anchor>transformers.onnx.OnnxConfigWithPast.fill_with_past_key_values_</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L552</source><parameters>[{"name": "inputs_or_outputs", "val": ": Mapping"}, {"name": "direction", "val": ": str"}, {"name": "inverted_values_shape", "val": ": bool = False"}]</parameters><paramsdesc>- **inputs_or_outputs** -- The mapping to fill.
- **direction** -- either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
  output mapping, this is important for axes naming.
- **inverted_values_shape** --
  If `True`, store values on dynamic axis 1, else on axis 2.</paramsdesc><paramgroups>0</paramgroups></docstring>

Fill the input_or_outputs mapping with past_key_values dynamic axes considering.




</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>with_past</name><anchor>transformers.onnx.OnnxConfigWithPast.with_past</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L454</source><parameters>[{"name": "config", "val": ": PretrainedConfig"}, {"name": "task", "val": ": str = 'default'"}]</parameters><paramsdesc>- **config** -- The underlying model's config to use when exporting to ONNX</paramsdesc><paramgroups>0</paramgroups><retdesc>OnnxConfig with `.use_past = True`</retdesc></docstring>

Instantiate a OnnxConfig with `use_past` attribute set to True






</div></div>

### OnnxSeq2SeqConfigWithPast[[transformers.onnx.OnnxSeq2SeqConfigWithPast]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.onnx.OnnxSeq2SeqConfigWithPast</name><anchor>transformers.onnx.OnnxSeq2SeqConfigWithPast</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/config.py#L592</source><parameters>[{"name": "config", "val": ": PretrainedConfig"}, {"name": "task", "val": ": str = 'default'"}, {"name": "patching_specs", "val": ": typing.Optional[list[transformers.onnx.config.PatchingSpec]] = None"}, {"name": "use_past", "val": ": bool = False"}]</parameters></docstring>


</div>

## ONNX Features

Each ONNX configuration is associated with a set of _features_ that enable you
to export models for different types of topologies or tasks.

### FeaturesManager[[transformers.onnx.FeaturesManager]]

<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>class transformers.onnx.FeaturesManager</name><anchor>transformers.onnx.FeaturesManager</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L71</source><parameters>[]</parameters></docstring>



<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>check_supported_model_or_raise</name><anchor>transformers.onnx.FeaturesManager.check_supported_model_or_raise</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L599</source><parameters>[{"name": "model", "val": ": PreTrainedModel"}, {"name": "feature", "val": ": str = 'default'"}]</parameters><paramsdesc>- **model** -- The model to export.
- **feature** -- The name of the feature to check if it is available.</paramsdesc><paramgroups>0</paramgroups><retdesc>(str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties.</retdesc></docstring>

Check whether or not the model has the requested features.






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_config</name><anchor>transformers.onnx.FeaturesManager.get_config</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L622</source><parameters>[{"name": "model_type", "val": ": str"}, {"name": "feature", "val": ": str"}]</parameters><paramsdesc>- **model_type** (`str`) --
  The model type to retrieve the config for.
- **feature** (`str`) --
  The feature to retrieve the config for.</paramsdesc><paramgroups>0</paramgroups><rettype>`OnnxConfig`</rettype><retdesc>config for the combination</retdesc></docstring>

Gets the OnnxConfig for a model_type and feature combination.








</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_model_class_for_feature</name><anchor>transformers.onnx.FeaturesManager.get_model_class_for_feature</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L559</source><parameters>[{"name": "feature", "val": ": str"}]</parameters><paramsdesc>- **feature** (`str`) --
  The feature required.</paramsdesc><paramgroups>0</paramgroups><retdesc>The AutoModel class corresponding to the feature.</retdesc></docstring>

Attempts to retrieve an AutoModel class from a feature name.






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_model_from_feature</name><anchor>transformers.onnx.FeaturesManager.get_model_from_feature</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L580</source><parameters>[{"name": "feature", "val": ": str"}, {"name": "model", "val": ": str"}, {"name": "cache_dir", "val": ": typing.Optional[str] = None"}]</parameters><paramsdesc>- **feature** (`str`) --
  The feature required.
- **model** (`str`) --
  The name of the model to export.</paramsdesc><paramgroups>0</paramgroups><retdesc>The instance of the model.</retdesc></docstring>

Attempts to retrieve a model from a model's name and the feature to be enabled.






</div>
<div class="docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8">


<docstring><name>get_supported_features_for_model_type</name><anchor>transformers.onnx.FeaturesManager.get_supported_features_for_model_type</anchor><source>https://github.com/huggingface/transformers/blob/main/src/transformers/onnx/features.py#L529</source><parameters>[{"name": "model_type", "val": ": str"}, {"name": "model_name", "val": ": typing.Optional[str] = None"}]</parameters><paramsdesc>- **model_type** (`str`) --
  The model type to retrieve the supported features for.
- **model_name** (`str`, *optional*) --
  The name attribute of the model object, only used for the exception message.</paramsdesc><paramgroups>0</paramgroups><retdesc>The dictionary mapping each feature to a corresponding OnnxConfig constructor.</retdesc></docstring>

Tries to retrieve the feature -> OnnxConfig constructor map from the model type.






</div></div>

<EditOnGithub source="https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/onnx.md" />
