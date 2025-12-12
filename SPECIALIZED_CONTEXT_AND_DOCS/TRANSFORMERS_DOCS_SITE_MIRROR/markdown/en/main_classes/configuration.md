# Configuration

The base class [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) implements the common methods for loading/saving a configuration
either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded
from HuggingFace’s AWS S3 repository).

Each derived config class implements model specific attributes. Common attributes present in all config classes are:
`hidden_size`, `num_attention_heads`, and `num_hidden_layers`. Text models further implement:
`vocab_size`.

## PreTrainedConfig

### class transformers.PreTrainedConfig

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L53)

( output\_hidden\_states: bool = False output\_attentions: bool = False return\_dict: bool = True dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None tie\_word\_embeddings: bool = True chunk\_size\_feed\_forward: int = 0 is\_encoder\_decoder: bool = False is\_decoder: bool = False cross\_attention\_hidden\_size: typing.Optional[int] = None add\_cross\_attention: bool = False tie\_encoder\_decoder: bool = False architectures: typing.Optional[list[str]] = None finetuning\_task: typing.Optional[str] = None id2label: typing.Optional[dict[int, str]] = None label2id: typing.Optional[dict[str, int]] = None num\_labels: typing.Optional[int] = None task\_specific\_params: typing.Optional[dict[str, typing.Any]] = None problem\_type: typing.Optional[str] = None tokenizer\_class: typing.Optional[str] = None prefix: typing.Optional[str] = None bos\_token\_id: typing.Optional[int] = None pad\_token\_id: typing.Optional[int] = None eos\_token\_id: typing.Optional[int] = None sep\_token\_id: typing.Optional[int] = None decoder\_start\_token\_id: typing.Optional[int] = None \*\*kwargs  )

Parameters

* **name\_or\_path** (`str`, *optional*, defaults to `""`) —
  Store the string that was passed to [PreTrainedModel.from\_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) as `pretrained_model_name_or_path`
  if the configuration was created with such a method.
* **output\_hidden\_states** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should return all hidden-states.
* **output\_attentions** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should returns all attentions.
* **return\_dict** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether the model is used as an encoder/decoder or not.
* **is\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether to only use the decoder in an encoder-decoder architecture, otherwise it has no effect on
  decoder-only or encoder-only architectures.
* **cross\_attention\_hidden\_size** (`bool`, *optional*) —
  The hidden size of the cross-attention layer in case the model is used as a decoder in an encoder-decoder
  setting and the cross-attention hidden dimension differs from `self.config.hidden_size`.
* **add\_cross\_attention** (`bool`, *optional*, defaults to `False`) —
  Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
  that can be used as decoder models within the [EncoderDecoderModel](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) class, which consists of all models
  in `AUTO_MODELS_FOR_CAUSAL_LM`.
* **tie\_encoder\_decoder** (`bool`, *optional*, defaults to `False`) —
  Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
  and decoder model to have the exact same parameter names.
* **chunk\_size\_feed\_forward** (`int`, *optional*, defaults to `0`) —
  The chunk size of all feed forward layers in the residual attention blocks. A chunk size of `0` means that
  the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes `n` <
  sequence\_length embeddings at a time. For more information on feed forward chunking, see [How does Feed
  Forward Chunking work?](../glossary.html#feed-forward-chunking).

Parameters for fine-tuning tasks

* **architectures** (`list[str]`, *optional*) —
  Model architectures that can be used with the model pretrained weights.
* **finetuning\_task** (`str`, *optional*) —
  Name of the task used to fine-tune the model.
* **id2label** (`dict[int, str]`, *optional*) —
  A map from index (for instance prediction index, or target index) to label.
* **label2id** (`dict[str, int]`, *optional*) —
  A map from label to index for the model.
* **num\_labels** (`int`, *optional*) —
  Number of labels to use in the last layer added to the model, typically for a classification task.
* **task\_specific\_params** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments to store for the current task.
* **problem\_type** (`str`, *optional*) —
  Problem type for `XxxForSequenceClassification` models. Can be one of `"regression"`,
  `"single_label_classification"` or `"multi_label_classification"`.

Parameters linked to the tokenizer

* **tokenizer\_class** (`str`, *optional*) —
  The name of the associated tokenizer class to use (if none is set, will use the tokenizer associated to the
  model by default).
* **prefix** (`str`, *optional*) —
  A specific prompt that should be added at the beginning of each text before calling the model.
* **bos\_token\_id** (`int`, *optional*) —
  The id of the *beginning-of-stream* token.
* **pad\_token\_id** (`int`, *optional*) —
  The id of the *padding* token.
* **eos\_token\_id** (`int`, *optional*) —
  The id of the *end-of-stream* token.
* **decoder\_start\_token\_id** (`int`, *optional*) —
  If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
* **sep\_token\_id** (`int`, *optional*) —
  The id of the *separation* token.

PyTorch specific parameters

* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether the model’s input and output word embeddings should be tied. Note that this is only relevant if the
  model has a output word embedding layer.
* **dtype** (`str`, *optional*) —
  The `dtype` of the weights. This attribute can be used to initialize the model to a non-default `dtype`
  (which is normally `float32`) and thus allow for optimal storage allocation. For example, if the saved
  model is `float16`, ideally we want to load it back using the minimal amount of memory needed to load
  `float16` weights.

Base class for all configuration classes. Handles a few parameters common to all models’ configurations as well as
methods for loading/downloading/saving configurations.

> A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
> initialize a model does **not** load the model weights. It only affects the model’s configuration.

Class attributes (overridden by derived classes):

* **model\_type** (`str`) — An identifier for the model type, serialized into the JSON file, and used to recreate
  the correct object in [AutoConfig](/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig).
* **has\_no\_defaults\_at\_init** (`bool`) — Whether the config class can be initialized without providing input arguments.
  Some configurations requires inputs to be defined at init and have no default values, usually these are composite configs,
  (but not necessarily) such as [EncoderDecoderConfig](/docs/transformers/main/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) or [~RagConfig](/docs/transformers/main/en/model_doc/rag#transformers.RagConfig). They have to be initialized from
  two or more configs of type [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig).
* **keys\_to\_ignore\_at\_inference** (`list[str]`) — A list of keys to ignore by default when looking at dictionary
  outputs of the model during inference.
* **attribute\_map** (`dict[str, str]`) — A dict that maps model specific attribute names to the standardized
  naming of attributes.
* **base\_model\_tp\_plan** (`dict[str, Any]`) — A dict that maps sub-modules FQNs of a base model to a tensor
  parallel plan applied to the sub-module when `model.tensor_parallel` is called.
* **base\_model\_pp\_plan** (`dict[str, tuple[list[str]]]`) — A dict that maps child-modules of a base model to a
  pipeline parallel plan that enables users to place the child-module on the appropriate device.

Common attributes (present in all subclasses):

* **vocab\_size** (`int`) — The number of tokens in the vocabulary, which is also the first dimension of the
  embeddings matrix (this attribute may be missing for models that don’t have a text modality like ViT).
* **hidden\_size** (`int`) — The hidden size of the model.
* **num\_attention\_heads** (`int`) — The number of attention heads used in the multi-head attention layers of the
  model.
* **num\_hidden\_layers** (`int`) — The number of blocks in the model.

> Setting parameters for sequence generation in the model config is deprecated. For backward compatibility, loading
> some of them will still be possible, but attempting to overwrite them will throw an exception — you should set
> them in a [~transformers.GenerationConfig]. Check the documentation of [~transformers.GenerationConfig] for more
> information about the individual parameters.

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711)

( repo\_id: str commit\_message: str | None = None commit\_description: str | None = None private: bool | None = None token: bool | str | None = None revision: str | None = None create\_pr: bool = False max\_shard\_size: int | str | None = '50GB' tags: list[str] | None = None  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your config to. It should contain your organization name
  when pushing to a given organization.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload config"`.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"50GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`).
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the configuration file to the 🤗 Model Hub.

Examples:

```
from transformers import AutoConfig

config = AutoConfig.from_pretrained("google-bert/bert-base-cased")

# Push the config to your namespace with the name "my-finetuned-bert".
config.push_to_hub("my-finetuned-bert")

# Push the config to an organization with the name "my-finetuned-bert".
config.push_to_hub("huggingface/my-finetuned-bert")
```

#### dict\_dtype\_to\_str

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L1000)

( d: dict  )

Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it’s not None,
converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *“float32”*
string, which can then be stored in the json format.

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L725)

( config\_dict: dict \*\*kwargs  ) → [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

Parameters

* **config\_dict** (`dict[str, Any]`) —
  Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
  retrieved from a pretrained checkpoint by leveraging the [get\_config\_dict()](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.get_config_dict) method.
* **kwargs** (`dict[str, Any]`) —
  Additional parameters from which to initialize the configuration object.

Returns

[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

The configuration object instantiated from those parameters.

Instantiates a [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) from a Python dictionary of parameters.

#### from\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L794)

( json\_file: str | os.PathLike  ) → [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

Parameters

* **json\_file** (`str` or `os.PathLike`) —
  Path to the JSON file containing the parameters.

Returns

[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

The configuration object instantiated from that JSON file.

Instantiates a [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) from the path to a JSON file of parameters.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L493)

( pretrained\_model\_name\_or\_path: str | os.PathLike cache\_dir: str | os.PathLike | None = None force\_download: bool = False local\_files\_only: bool = False token: str | bool | None = None revision: str = 'main' \*\*kwargs  ) → [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained model configuration hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a configuration file saved using the
    [save\_pretrained()](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.save_pretrained) method, e.g., `./my_model_directory/`.
  + a path or url to a saved configuration JSON *file*, e.g., `./my_model_directory/configuration.json`.
* **cache\_dir** (`str` or `os.PathLike`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force to (re-)download the configuration files and override the cached versions if
  they exist.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
* **token** (`str` or `bool`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.
* **return\_unused\_kwargs** (`bool`, *optional*, defaults to `False`) —
  If `False`, then this function returns just the final configuration object.

  If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused\_kwargs* is a
  dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
  part of `kwargs` which has not been used to update `config` and is otherwise ignored.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
* **kwargs** (`dict[str, Any]`, *optional*) —
  The values in kwargs of any keys which are configuration attributes will be used to override the loaded
  values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
  by the `return_unused_kwargs` keyword parameter.

Returns

[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)

The configuration object instantiated from this pretrained model.

Instantiate a [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) (or a derived class) from a pretrained model configuration.

Examples:

```
# We can't instantiate directly the base class *PreTrainedConfig* so let's show the examples on a
# derived class: BertConfig
config = BertConfig.from_pretrained(
    "google-bert/bert-base-uncased"
)  # Download configuration from huggingface.co and cache.
config = BertConfig.from_pretrained(
    "./test/saved_model/"
)  # E.g. config (or model) was saved using *save_pretrained('./test/saved_model/')*
config = BertConfig.from_pretrained("./test/saved_model/my_configuration.json")
config = BertConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
assert config.output_attentions == True
config, unused_kwargs = BertConfig.from_pretrained(
    "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
)
assert config.output_attentions == True
assert unused_kwargs == {"foo": False}
```

#### get\_config\_dict

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L601)

( pretrained\_model\_name\_or\_path: str | os.PathLike \*\*kwargs  ) → `tuple[Dict, Dict]`

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

Returns

`tuple[Dict, Dict]`

The dictionary(ies) that will be used to instantiate the configuration object.

From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
[PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) using `from_dict`.

#### get\_text\_config

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L1119)

( decoder = None encoder = None  )

Parameters

* **decoder** (`Optional[bool]`, *optional*) —
  If set to `True`, then only search for decoder config names.
* **encoder** (`Optional[bool]`, *optional*) —
  If set to `True`, then only search for encoder config names.

Returns the text config related to the text input (encoder) or text output (decoder) of the model. The
`decoder` and `encoder` input arguments can be used to specify which end of the model we are interested in,
which is useful on models that have both text input and output modalities.

There are three possible outcomes of using this method:

1. On most models, it returns the original config instance itself.
2. On newer (2024+) composite models, it returns the text section of the config, which is nested under a set
   of valid names.
3. On older (2023-) composite models, it discards decoder-only parameters when `encoder=True` and vice-versa.

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L1044)

( auto\_class = 'AutoConfig'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoConfig"`) —
  The auto class to register this new configuration with.

Register this class with a given auto class. This should only be used for custom configurations as the ones in
the library are already mapped with `AutoConfig`.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L434)

( save\_directory: str | os.PathLike push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the configuration JSON file will be saved (will be created if it does not exist).
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
[from\_pretrained()](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.from_pretrained) class method.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L886)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance.

Serializes this instance to a Python dictionary.

#### to\_diff\_dict

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L827)

( ) → dict[str, Any]

Returns

dict[str, Any]

Dictionary of all the attributes that make up this configuration instance.

Removes all attributes from the configuration that correspond to the default config attributes for
better readability, while always retaining the `config` attribute from the class. Serializes to a
Python dictionary.

#### to\_json\_file

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L938)

( json\_file\_path: str | os.PathLike use\_diff: bool = True  )

Parameters

* **json\_file\_path** (`str` or `os.PathLike`) —
  Path to the JSON file in which this configuration instance’s parameters will be saved.
* **use\_diff** (`bool`, *optional*, defaults to `True`) —
  If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
  is serialized to JSON file.

Save this instance to a JSON file.

#### to\_json\_string

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L920)

( use\_diff: bool = True  ) → `str`

Parameters

* **use\_diff** (`bool`, *optional*, defaults to `True`) —
  If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
  is serialized to JSON string.

Returns

`str`

String containing all the attributes that make up this configuration instance in JSON format.

Serializes this instance to a JSON string.

#### update

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L952)

( config\_dict: dict  )

Parameters

* **config\_dict** (`dict[str, Any]`) — Dictionary of attributes that should be updated for this class.

Updates attributes of this class with attributes from `config_dict`.

#### update\_from\_string

 [< source >](https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L962)

( update\_str: str  )

Parameters

* **update\_str** (`str`) — String with attributes that should be updated for this class.

Updates attributes of this class with attributes from `update_str`.

The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
“n\_embd=10,resid\_pdrop=0.2,scale\_attn\_weights=false,summary\_type=cls\_index”

The keys to change have to already exist in the config object.

 [Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/configuration.md)
