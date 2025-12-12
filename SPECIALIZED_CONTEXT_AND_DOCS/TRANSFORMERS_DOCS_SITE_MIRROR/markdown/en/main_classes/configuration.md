# Configuration

The base class [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) implements the common methods for loading/saving a configuration
either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded
from HuggingFace's AWS S3 repository).

Each derived config class implements model specific attributes. Common attributes present in all config classes are:
`hidden_size`, `num_attention_heads`, and `num_hidden_layers`. Text models further implement:
`vocab_size`.

## PreTrainedConfig[[transformers.PreTrainedConfig]]

#### transformers.PreTrainedConfig[[transformers.PreTrainedConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L53)

Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
methods for loading/downloading/saving configurations.

A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
initialize a model does **not** load the model weights. It only affects the model's configuration.

Class attributes (overridden by derived classes):

- **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, and used to recreate
  the correct object in [AutoConfig](/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig).
- **has_no_defaults_at_init** (`bool`) -- Whether the config class can be initialized without providing input arguments.
  Some configurations requires inputs to be defined at init and have no default values, usually these are composite configs,
  (but not necessarily) such as [EncoderDecoderConfig](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) or [~RagConfig](/docs/transformers/main/en/model_doc/rag#transformers.RagConfig). They have to be initialized from
  two or more configs of type [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig).
- **keys_to_ignore_at_inference** (`list[str]`) -- A list of keys to ignore by default when looking at dictionary
  outputs of the model during inference.
- **attribute_map** (`dict[str, str]`) -- A dict that maps model specific attribute names to the standardized
  naming of attributes.
- **base_model_tp_plan** (`dict[str, Any]`) -- A dict that maps sub-modules FQNs of a base model to a tensor
  parallel plan applied to the sub-module when `model.tensor_parallel` is called.
- **base_model_pp_plan** (`dict[str, tuple[list[str]]]`) -- A dict that maps child-modules of a base model to a
  pipeline parallel plan that enables users to place the child-module on the appropriate device.

Common attributes (present in all subclasses):

- **vocab_size** (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
  embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
- **hidden_size** (`int`) -- The hidden size of the model.
- **num_attention_heads** (`int`) -- The number of attention heads used in the multi-head attention layers of the
  model.
- **num_hidden_layers** (`int`) -- The number of blocks in the model.

Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading
some of them will still be possible, but attempting to overwrite them will throw an exception -- you should set
them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more
information about the individual parameters.

push_to_hubtransformers.PreTrainedConfig.push_to_hubhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711[{"name": "repo_id", "val": ": str"}, {"name": "commit_message", "val": ": str | None = None"}, {"name": "commit_description", "val": ": str | None = None"}, {"name": "private", "val": ": bool | None = None"}, {"name": "token", "val": ": bool | str | None = None"}, {"name": "revision", "val": ": str | None = None"}, {"name": "create_pr", "val": ": bool = False"}, {"name": "max_shard_size", "val": ": int | str | None = '50GB'"}, {"name": "tags", "val": ": list[str] | None = None"}]- **repo_id** (`str`) --
  The name of the repository you want to push your config to. It should contain your organization name
  when pushing to a given organization.
- **commit_message** (`str`, *optional*) --
  Message to commit while pushing. Will default to `"Upload config"`.
- **commit_description** (`str`, *optional*) --
  The description of the commit that will be created
- **private** (`bool`, *optional*) --
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
- **token** (`bool` or `str`, *optional*) --
  The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
- **revision** (`str`, *optional*) --
  Branch to push the uploaded files to.
- **create_pr** (`bool`, *optional*, defaults to `False`) --
  Whether or not to create a PR with the uploaded files or directly commit.
- **max_shard_size** (`int` or `str`, *optional*, defaults to `"50GB"`) --
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`).
- **tags** (`list[str]`, *optional*) --
  List of tags to push on the Hub.0

Upload the configuration file to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("google-bert/bert-base-cased")

# Push the config to your namespace with the name "my-finetuned-bert".
config.push_to_hub("my-finetuned-bert")

# Push the config to an organization with the name "my-finetuned-bert".
config.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

name_or_path (`str`, *optional*, defaults to `""`) : Store the string that was passed to [PreTrainedModel.from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) as `pretrained_model_name_or_path` if the configuration was created with such a method.

output_hidden_states (`bool`, *optional*, defaults to `False`) : Whether or not the model should return all hidden-states.

output_attentions (`bool`, *optional*, defaults to `False`) : Whether or not the model should returns all attentions.

return_dict (`bool`, *optional*, defaults to `True`) : Whether or not the model should return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

is_encoder_decoder (`bool`, *optional*, defaults to `False`) : Whether the model is used as an encoder/decoder or not.

is_decoder (`bool`, *optional*, defaults to `False`) : Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on decoder-only or encoder-only architectures.

cross_attention_hidden_size (`bool`, *optional*) : The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.

add_cross_attention (`bool`, *optional*, defaults to `False`) : Whether cross-attention layers should be added to the model. Note, this option is only relevant for models that can be used as decoder models within the [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) class, which consists of all models in `AUTO_MODELS_FOR_CAUSAL_LM`.

tie_encoder_decoder (`bool`, *optional*, defaults to `False`) : Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder and decoder model to have the exact same parameter names.

chunk_size_feed_forward (`int`, *optional*, defaults to `0`) : The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` "`.   

return_unused_kwargs (`bool`, *optional*, defaults to `False`) : If `False`, then this function returns just the final configuration object.  If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of `kwargs` which has not been used to update `config` and is otherwise ignored.

subfolder (`str`, *optional*, defaults to `""`) : In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can specify the folder name here.

kwargs (`dict[str, Any]`, *optional*) : The values in kwargs of any keys which are configuration attributes will be used to override the loaded values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

**Returns:**

`[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)`

The configuration object instantiated from this pretrained model.
#### get_config_dict[[transformers.PreTrainedConfig.get_config_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L601)

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) using `from_dict`.

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`) : The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

**Returns:**

``tuple[Dict, Dict]``

The dictionary(ies) that will be used to instantiate the configuration object.
#### get_text_config[[transformers.PreTrainedConfig.get_text_config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L1119)

Returns the text config related to the text input (encoder) or text output (decoder) of the model. The
`decoder` and `encoder` input arguments can be used to specify which end of the model we are interested in,
which is useful on models that have both text input and output modalities.

There are three possible outcomes of using this method:
1. On most models, it returns the original config instance itself.
2. On newer (2024+) composite models, it returns the text section of the config, which is nested under a set
   of valid names.
3. On older (2023-) composite models, it discards decoder-only parameters when `encoder=True` and vice-versa.

**Parameters:**

decoder (`Optional[bool]`, *optional*) : If set to `True`, then only search for decoder config names.

encoder (`Optional[bool]`, *optional*) : If set to `True`, then only search for encoder config names.
#### register_for_auto_class[[transformers.PreTrainedConfig.register_for_auto_class]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L1044)

Register this class with a given auto class. This should only be used for custom configurations as the ones in
the library are already mapped with `AutoConfig`.

**Parameters:**

auto_class (`str` or `type`, *optional*, defaults to `"AutoConfig"`) : The auto class to register this new configuration with.
#### save_pretrained[[transformers.PreTrainedConfig.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L434)

Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
[from_pretrained()](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.from_pretrained) class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory where the configuration JSON file will be saved (will be created if it does not exist).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### to_dict[[transformers.PreTrainedConfig.to_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L886)

Serializes this instance to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this configuration instance.
#### to_diff_dict[[transformers.PreTrainedConfig.to_diff_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L827)

Removes all attributes from the configuration that correspond to the default config attributes for
better readability, while always retaining the `config` attribute from the class. Serializes to a
Python dictionary.

**Returns:**

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance.
#### to_json_file[[transformers.PreTrainedConfig.to_json_file]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L938)

Save this instance to a JSON file.

**Parameters:**

json_file_path (`str` or `os.PathLike`) : Path to the JSON file in which this configuration instance's parameters will be saved.

use_diff (`bool`, *optional*, defaults to `True`) : If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()` is serialized to JSON file.
#### to_json_string[[transformers.PreTrainedConfig.to_json_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L920)

Serializes this instance to a JSON string.

**Parameters:**

use_diff (`bool`, *optional*, defaults to `True`) : If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()` is serialized to JSON string.

**Returns:**

``str``

String containing all the attributes that make up this configuration instance in JSON format.
#### update[[transformers.PreTrainedConfig.update]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L952)

Updates attributes of this class with attributes from `config_dict`.

**Parameters:**

config_dict (`dict[str, Any]`) : Dictionary of attributes that should be updated for this class.
#### update_from_string[[transformers.PreTrainedConfig.update_from_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L962)

Updates attributes of this class with attributes from `update_str`.

The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
"n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

The keys to change have to already exist in the config object.

**Parameters:**

update_str (`str`) : String with attributes that should be updated for this class.
