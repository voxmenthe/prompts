# Models

The base class [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) implements the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace’s Hub).

[PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) also implements a few methods which are common among all the models to:

* resize the input token embeddings when new tokens are added to the vocabulary
* prune the attention heads of the model.

The other methods that are common to each model are defined in [ModuleUtilsMixin](/docs/transformers/v4.56.2/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin) and [GenerationMixin](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin).

## PreTrainedModel

### class transformers.PreTrainedModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2003)

( config: PretrainedConfig \*inputs \*\*kwargs  )

Base class for all models.

[PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel) takes care of storing the configuration of the models and handles methods for loading,
downloading and saving models as well as a few methods common to all models to:

* resize the input embeddings,
* prune heads in the self-attention heads.

Class attributes (overridden by derived classes):

* **config\_class** ([PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig)) — A subclass of [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) to use as configuration class
  for this model architecture.
* **load\_tf\_weights** (`Callable`) — A python *method* for loading a TensorFlow checkpoint in a PyTorch model,
  taking as arguments:

  + **model** ([PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)) — An instance of the model on which to load the TensorFlow checkpoint.
  + **config** (`PreTrainedConfig`) — An instance of the configuration associated to the model.
  + **path** (`str`) — A path to the TensorFlow checkpoint.
* **base\_model\_prefix** (`str`) — A string indicating the attribute associated to the base model in derived
  classes of the same architecture adding modules on top of the base model.
* **is\_parallelizable** (`bool`) — A flag indicating whether this model supports model parallelization.
* **main\_input\_name** (`str`) — The name of the principal input to the model (often `input_ids` for NLP
  models, `pixel_values` for vision models and `input_values` for speech models).
* **can\_record\_outputs** (dict):

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847)

( repo\_id: str use\_temp\_dir: typing.Optional[bool] = None commit\_message: typing.Optional[str] = None private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None max\_shard\_size: typing.Union[str, int, NoneType] = '5GB' create\_pr: bool = False safe\_serialization: bool = True revision: typing.Optional[str] = None commit\_description: typing.Optional[str] = None tags: typing.Optional[list[str]] = None \*\*deprecated\_kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your model to. It should contain your organization name
  when pushing to a given organization.
* **use\_temp\_dir** (`bool`, *optional*) —
  Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
  Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload model"`.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
  is not specified.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
  Google Colab instances without any CPU OOM issues.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to convert the model weights in safetensors format for safer serialization.
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the model file to the 🤗 Model Hub.

Examples:


```
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")

# Push the model to your namespace with the name "my-finetuned-bert".
model.push_to_hub("my-finetuned-bert")

# Push the model to an organization with the name "my-finetuned-bert".
model.push_to_hub("huggingface/my-finetuned-bert")
```

#### add\_model\_tags

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2362)

( tags: typing.Union[list[str], str]  )

Parameters

* **tags** (`Union[list[str], str]`) —
  The desired tags to inject in the model

Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
not overwrite existing tags in the model.

Examples:


```
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")

model.add_model_tags(["custom", "custom-bert"])

# Push the model to your namespace with the name "my-custom-bert".
model.push_to_hub("my-custom-bert")
```

#### can\_generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2478)

( ) → `bool`

Returns

`bool`

Whether this model can generate sequences with `.generate()`.

Returns whether this model can generate sequences with `.generate()` from the `GenerationMixin`.

Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
for instance, the model instance will have a populated `generation_config` attribute.

#### dequantize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2344)

( )

Potentially dequantize the model in case it has been quantized by a quantization method that support
dequantization.

#### disable\_input\_require\_grads

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2981)

( )

Removes the `_require_grads_hook`.

#### enable\_input\_require\_grads

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2970)

( )

Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
the model weights fixed.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L4501)

( pretrained\_model\_name\_or\_path: typing.Union[str, os.PathLike, NoneType] \*model\_args config: typing.Union[transformers.configuration\_utils.PretrainedConfig, str, os.PathLike, NoneType] = None cache\_dir: typing.Union[str, os.PathLike, NoneType] = None ignore\_mismatched\_sizes: bool = False force\_download: bool = False local\_files\_only: bool = False token: typing.Union[bool, str, NoneType] = None revision: str = 'main' use\_safetensors: typing.Optional[bool] = None weights\_only: bool = True \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`, *optional*) —
  Can be either:
  + A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing model weights saved using
    [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`.
  + A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    this case, `from_tf` should be set to `True` and a configuration object should be provided as
    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
  + A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
    `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
    `True`.
  + `None` if you are both providing the configuration and state dictionary (resp. with keyword
    arguments `config` and `state_dict`).
* **model\_args** (sequence of positional arguments, *optional*) —
  All remaining positional arguments will be passed to the underlying model’s `__init__` method.
* **config** (`Union[PretrainedConfig, str, os.PathLike]`, *optional*) —
  Can be either:
  + an instance of a class derived from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig),
  + a string or path valid as input to [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained).

  Configuration for the model to use instead of an automatically loaded configuration. Configuration can
  be automatically loaded when:

  + The model is a model provided by the library (loaded with the *model id* string of a pretrained
    model).
  + The model was saved using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the
    save directory.
  + The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    configuration JSON file named *config.json* is found in the directory.
* **state\_dict** (`dict[str, torch.Tensor]`, *optional*) —
  A state dictionary to use instead of a state dictionary loaded from saved weights file.

  This option can be used if you want to create a model from a pretrained configuration but load your own
  weights. In this case though, you should check if using [save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.
* **cache\_dir** (`Union[str, os.PathLike]`, *optional*) —
  Path to a directory in which a downloaded pretrained model configuration should be cached if the
  standard cache should not be used.
* **from\_tf** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a TensorFlow checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **from\_flax** (`bool`, *optional*, defaults to `False`) —
  Load the model weights from a Flax checkpoint save file (see docstring of
  `pretrained_model_name_or_path` argument).
* **ignore\_mismatched\_sizes** (`bool`, *optional*, defaults to `False`) —
  Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
  as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
  checkpoint with 3 labels).
* **force\_download** (`bool`, *optional*, defaults to `False`) —
  Whether or not to force the (re-)download of the model weights and configuration files, overriding the
  cached versions if they exist.
* **resume\_download** —
  Deprecated and ignored. All downloads are now resumed by default when possible.
  Will be removed in v5 of Transformers.
* **proxies** (`dict[str, str]`, *optional*) —
  A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
* **output\_loading\_info(`bool`,** *optional*, defaults to `False`) —
  Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
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
* **attn\_implementation** (`str`, *optional*) —
  The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

  Accept HF kernel references in the form:

  /[@][:]
  + and  are any non-"/" and non-":" sequences.
  + “@” is optional (branch, tag, or commit-ish), e.g. “@main”, “@v1.2.0”, “@abc123”.
  + ”:” is optional and selects a function inside the kernel repo.
  + Both options can appear together and in this order only: @revision first, then :kernel\_name.
  + We intentionally allow a leading ”|” prefix (e.g., “flash|…”) because the code
    strips it before loading; ’|’ is not excluded in the character classes here.

  Examples that match:
  “org/model”
  “org/model@main”
  “org/model:custom\_kernel”
  “org/[model@v1.2.3](mailto:model@v1.2.3):custom\_kernel”

Parameters for big model inference

* **dtype** (`str` or `torch.dtype`, *optional*) —
  Override the default `torch_dtype` and load the model under a specific `dtype`. The different options
  are:
  1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
     `dtype`, ignoring the model’s `config.dtype` if one exists. If not specified

     + the model will get loaded in `torch.float` (fp32).
  2. `"auto"` - A `dtype` or `torch_dtype` entry in the `config.json` file of the model will be
     attempted to be used. If this entry isn’t found then next check the `dtype` of the first weight in
     the checkpoint that’s of a floating point type and use that as `dtype`. This will load the model
     using the `dtype` it was saved in at the end of the training. It can’t be used as an indicator of how
     the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
  3. A string that is a valid `torch.dtype`. E.g. “float32” loads the model in `torch.float32`, “float16” loads in `torch.float16` etc.

  For some models the `dtype` they were trained in is unknown - you may try to check the model’s paper or
  reach out to the authors and ask them to add this information to the model’s card and to insert the
  `dtype` or `torch_dtype` entry in `config.json` on the hub.
* **device\_map** (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*) —
  A map that specifies where each submodule should go. It doesn’t need to be refined to each
  parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
  same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
  like `1`) on which the model will be allocated, the device map will map the entire model to this
  device. Passing `device_map = 0` means put the whole model on GPU 0.

  To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
  more information about each option see [designing a device
  map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
* **max\_memory** (`Dict`, *optional*) —
  A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
  GPU and the available CPU RAM if unset.
* **tp\_plan** (`str`, *optional*) —
  A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Currently, it only accepts
  `tp_plan="auto"` to use predefined plan based on the model. Note that if you use it, you should launch your script accordingly with
  `torchrun [args] script.py`. This will be much faster than using a `device_map`, but has limitations.
* **tp\_size** (`str`, *optional*) —
  A torch tensor parallel degree. If not provided would default to world size.
* **device\_mesh** (`torch.distributed.DeviceMesh`, *optional*) —
  A torch device mesh. If not provided would default to world size. Used only for tensor parallel for now.
  If provided, it has to contain dimension named `"tp"` in case it’s > 1 dimensional, this dimension will be used for tensor parallelism
* **offload\_folder** (`str` or `os.PathLike`, *optional*) —
  If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
* **offload\_state\_dict** (`bool`, *optional*) —
  If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
  RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
  `True` when there is some disk offload.
* **offload\_buffers** (`bool`, *optional*) —
  Whether or not to offload the buffers with the model parameters.
* **quantization\_config** (`Union[QuantizationConfigMixin,Dict]`, *optional*) —
  A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
  bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
  `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
  quantizations and not preferred. consider inserting all such arguments into quantization\_config
  instead.
* **subfolder** (`str`, *optional*, defaults to `""`) —
  In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
  specify the folder name here.
* **variant** (`str`, *optional*) —
  If specified load weights from `variant` filename, *e.g.* pytorch\_model..bin. `variant` is
  ignored when using `from_tf` or `from_flax`.
* **use\_safetensors** (`bool`, *optional*, defaults to `None`) —
  Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
  is not installed, it will be set to `False`.
* **weights\_only** (`bool`, *optional*, defaults to `True`) —
  Indicates whether unpickler should be restricted to loading only tensors, primitive types,
  dictionaries and any types added via torch.serialization.add\_safe\_globals().
  When set to False, we can load wrapper tensor subclass weights.
* **key\_mapping** (`dict[str, str], *optional*) —
  A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
  architecture, but was not converted accordingly.
* **kwargs** (remaining dictionary of keyword arguments, *optional*) —
  Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
  `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
  automatically loaded:
  + If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    underlying model’s `__init__` method (we assume all relevant updates to the configuration have
    already been done)
  + If a configuration is not provided, `kwargs` will be first passed to the configuration class
    initialization function ([from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig.from_pretrained)). Each key of `kwargs` that
    corresponds to a configuration attribute will be used to override said attribute with the
    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    will be passed to the underlying model’s `__init__` function.

Instantiate a pretrained pytorch model from a pre-trained model configuration.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
the model, you should first set it back in training mode with `model.train()`.

The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
task.

The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
weights are discarded.

Activate the special [“offline-mode”](https://huggingface.co/transformers/installation.html#offline-mode) to
use this method in a firewalled environment.

Examples:


```
>>> from transformers import BertConfig, BertModel

>>> # Download model and configuration from huggingface.co and cache.
>>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
>>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
>>> model = BertModel.from_pretrained("./test/saved_model/")
>>> # Update configuration during loading.
>>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
>>> assert model.config.output_attentions == True
>>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
>>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
>>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
>>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
>>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
```

#### get\_compiled\_call

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L5972)

( compile\_config: typing.Optional[transformers.generation.configuration\_utils.CompileConfig]  )

Return a `torch.compile`‘d version of `self.__call__`. This is useful to dynamically choose between
non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don’t
want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
(where we want the speed-ups of compiled version with static shapes).

#### get\_decoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2987)

( )

Best-effort lookup of the *decoder* module.

Order of attempts (covers ~85 % of current usages):

1. `self.decoder`
2. `self.model` (many wrappers store the decoder here)
3. `self.model.get_decoder()` (nested wrappers)
4. fallback: raise for the few exotic models that need a bespoke rule

#### get\_memory\_footprint

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L4348)

( return\_buffers = True  )

Parameters

* **return\_buffers** (`bool`, *optional*, defaults to `True`) —
  Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
  are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
  norm layers. Please see: <https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2>

Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
PyTorch discussions: <https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2>

#### get\_parameter\_or\_buffer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L6070)

( target: str  )

Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
`get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.

#### gradient\_checkpointing\_disable

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3859)

( )

Deactivates gradient checkpointing for the current model.

Note that in other frameworks this feature can be referred to as “activation checkpointing” or “checkpoint
activations”.

#### gradient\_checkpointing\_enable

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3795)

( gradient\_checkpointing\_kwargs = None  )

Parameters

* **gradient\_checkpointing\_kwargs** (dict, *optional*) —
  Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.

Activates gradient checkpointing for the current model.

Note that in other frameworks this feature can be referred to as “activation checkpointing” or “checkpoint
activations”.

We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
the module. <https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2>

#### init\_weights

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3761)

( )

If needed prunes and maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
initialization logic in `_init_weights`.

#### initialize\_weights

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3076)

( )

This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
is extremely error prone and inefficient.

Note that the `torch.no_grad()` decorator is very important as well, as most of our `_init_weights` do not use
`torch.nn.init` functions (which are all no*grad by default), but simply do in-place ops such as
`module.weight.data.zero*()`.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2224)

( )

A method executed at the end of each Transformer model initialization, to execute code that needs the model’s
modules properly initialized (such as weight initialization).

This is also used when the user is running distributed code. We add hooks to the modules here, according to
the model’s tp\_plan!

#### prune\_heads

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3778)

( heads\_to\_prune: dict  )

Parameters

* **heads\_to\_prune** (`dict[int, list[int]]`) —
  Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
  to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
  layer 1 and heads 2 and 3 on layer 2.

Prunes heads of the base model.

#### register\_for\_auto\_class

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L5815)

( auto\_class = 'AutoModel'  )

Parameters

* **auto\_class** (`str` or `type`, *optional*, defaults to `"AutoModel"`) —
  The auto class to register this new model with.

Register this class with a given auto class. This should only be used for custom models as the ones in the
library are already mapped with an auto class.

#### resize\_token\_embeddings

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3294)

( new\_num\_tokens: typing.Optional[int] = None pad\_to\_multiple\_of: typing.Optional[int] = None mean\_resizing: bool = True  ) → `torch.nn.Embedding`

Parameters

* **new\_num\_tokens** (`int`, *optional*) —
  The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
  vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
  returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
  `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
  details about this, or help on choosing the correct value for resizing, refer to this guide:
  <https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc>
* **mean\_resizing** (`bool`) —
  Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings’ mean and
  covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.

  Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models,
  where the generated tokens’ probabilities won’t be affected by the added embeddings because initializing the new embeddings with the
  old embeddings’ mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings.
  Refer to this article for more information: <https://nlp.stanford.edu/~johnhew/vocab-expansion.html>

Returns

`torch.nn.Embedding`

Pointer to the input tokens Embeddings Module of the model.

Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

#### reverse\_bettertransformer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L5865)

( ) → [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)

Returns

[PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)

The model converted back to the original modeling.

Reverts the transformation from [to\_bettertransformer()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.to_bettertransformer) so that the original modeling is
used, for example in order to save the model.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3892)

( save\_directory: typing.Union[str, os.PathLike] is\_main\_process: bool = True state\_dict: typing.Optional[dict] = None save\_function: typing.Callable = <function save at 0x7f39a93c9900> push\_to\_hub: bool = False max\_shard\_size: typing.Union[int, str] = '5GB' safe\_serialization: bool = True variant: typing.Optional[str] = None token: typing.Union[bool, str, NoneType] = None save\_peft\_format: bool = True \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory to which to save. Will be created if it doesn’t exist.
* **is\_main\_process** (`bool`, *optional*, defaults to `True`) —
  Whether the process calling this is the main process or not. Useful when in distributed training like
  TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
  the main process to avoid race conditions.
* **state\_dict** (nested dictionary of `torch.Tensor`) —
  The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
  save parts of the model or if special precautions need to be taken when recovering the state dictionary
  of a model (like when using model parallelism).
* **save\_function** (`Callable`) —
  The function to use to save the state dictionary. Useful on distributed training like TPUs when one
  need to replace `torch.save` by another method.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
  lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
  We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
  without CPU OOM issues.

  If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
  which will be bigger than `max_shard_size`.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
* **variant** (`str`, *optional*) —
  If specified, weights are saved in the format pytorch\_model..bin.
* **token** (`str` or `bool`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
  the token generated when running `hf auth login` (stored in `~/.huggingface`).
* **save\_peft\_format** (`bool`, *optional*, defaults to `True`) —
  For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
  keys of the state dict of adapters needs to be prepended with `base_model.model`. Advanced users can
  disable this behaviours by setting `save_peft_format` to `False`.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Save a model and its configuration file to a directory, so that it can be re-loaded using the
[from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) class method.

#### set\_attn\_implementation

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L2869)

( attn\_implementation: typing.Union[str, dict]  )

Parameters

* **attn\_implementation** (`str` or `dict`) —
  The attention implementation to set for this model. It can be either a `str`, in which case it will be
  dispatched to all submodels if relevant, or a `dict` where keys are the sub\_configs name, in which case each
  submodel will dispatch the corresponding value.

Set the requested `attn_implementation` for this model.

#### set\_decoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3012)

( decoder  )

Symmetric setter. Mirrors the lookup logic used in `get_decoder`.

#### tie\_embeddings\_and\_encoder\_decoder

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3107)

( )

If set in the config, tie the weights between the input embeddings and the output embeddings,
and the encoder and decoder.

If the `torchscript` flag is set in the configuration, can’t handle parameter sharing so we are cloning the
weights instead.

#### tie\_weights

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L3131)

( )

Recursively (for all submodels) tie all the weights of the model.

#### to\_bettertransformer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L5837)

( ) → [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)

Returns

[PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel)

The model converted to BetterTransformer.

Converts the model to use [PyTorch’s native attention
implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to
Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a
subset of all Transformers models are supported.

PyTorch’s attention fastpath allows to speed up inference through kernel fusions and the use of [nested
tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog
post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).

#### warn\_if\_padding\_and\_no\_attention\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L5887)

( input\_ids attention\_mask  )

Shows a one-time warning if the input\_ids appear to contain padding and no attention mask was given.

Custom models should also include a `_supports_assign_param_buffer`, which determines if superfast init can apply
on the particular model. Signs that your model needs this are if `test_save_and_load_from_pretrained` fails. If so,
set this to `False`.

## ModuleUtilsMixin

### class transformers.modeling\_utils.ModuleUtilsMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1603)

( )

A few utilities for `torch.nn.Modules`, to be used as a mixin.

#### add\_memory\_hooks

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1634)

( )

Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

Increase in memory consumption is stored in a `mem_rss_diff` attribute for each module and can be reset to zero
with `model.reset_memory_hooks_state()`.

#### estimate\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1865)

( input\_dict: dict  ) → `int`

Parameters

* **inputs** (`dict`) — The model inputs.

Returns

`int`

The total number of tokens.

Helper function to estimate the total number of tokens from the model inputs.

#### floating\_point\_ops

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1886)

( input\_dict: dict exclude\_embeddings: bool = True  ) → `int`

Parameters

* **batch\_size** (`int`) —
  The batch size for the forward pass.
* **sequence\_length** (`int`) —
  The number of tokens in each line of the batch.
* **exclude\_embeddings** (`bool`, *optional*, defaults to `True`) —
  Whether or not to count embedding and softmax operations.

Returns

`int`

The number of floating-point operations.

Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
paper](https://huggingface.co/papers/2001.08361) section 2.1. Should be overridden for transformers with parameter
re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

#### get\_extended\_attention\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1721)

( attention\_mask: Tensor input\_shape: tuple device: device = None dtype: torch.float32 = None  )

Parameters

* **attention\_mask** (`torch.Tensor`) —
  Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
* **input\_shape** (`tuple[int]`) —
  The shape of the input to the model.

Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

#### get\_head\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1773)

( head\_mask: typing.Optional[torch.Tensor] num\_hidden\_layers: int is\_attention\_chunked: bool = False  )

Parameters

* **head\_mask** (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*) —
  The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
* **num\_hidden\_layers** (`int`) —
  The number of hidden layers in the model.
* **is\_attention\_chunked** (`bool`, *optional*, defaults to `False`) —
  Whether or not the attentions scores are computed by chunks or not.

Prepare the head mask if needed.

#### invert\_attention\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1670)

( encoder\_attention\_mask: Tensor  ) → `torch.Tensor`

Parameters

* **encoder\_attention\_mask** (`torch.Tensor`) — An attention mask.

Returns

`torch.Tensor`

The inverted attention mask.

Invert an attention mask (e.g., switches 0. and 1.).

#### num\_parameters

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1811)

( only\_trainable: bool = False exclude\_embeddings: bool = False  ) → `int`

Parameters

* **only\_trainable** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return only the number of trainable parameters
* **exclude\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to return only the number of non-embeddings parameters

Returns

`int`

The number of parameters.

Get number of (optionally, trainable or non-embeddings) parameters in the module.

#### reset\_memory\_hooks\_state

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L1646)

( )

Reset the `mem_rss_diff` attribute of each module (see [add\_memory\_hooks()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin.add_memory_hooks)).

## Pushing to the Hub

### class transformers.utils.PushToHubMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L722)

( )

A Mixin containing the functionality to push a model or tokenizer to the hub.

#### push\_to\_hub

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/hub.py#L847)

( repo\_id: str use\_temp\_dir: typing.Optional[bool] = None commit\_message: typing.Optional[str] = None private: typing.Optional[bool] = None token: typing.Union[bool, str, NoneType] = None max\_shard\_size: typing.Union[str, int, NoneType] = '5GB' create\_pr: bool = False safe\_serialization: bool = True revision: typing.Optional[str] = None commit\_description: typing.Optional[str] = None tags: typing.Optional[list[str]] = None \*\*deprecated\_kwargs  )

Parameters

* **repo\_id** (`str`) —
  The name of the repository you want to push your {object} to. It should contain your organization name
  when pushing to a given organization.
* **use\_temp\_dir** (`bool`, *optional*) —
  Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
  Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
* **commit\_message** (`str`, *optional*) —
  Message to commit while pushing. Will default to `"Upload {object}"`.
* **private** (`bool`, *optional*) —
  Whether to make the repo private. If `None` (default), the repo will be public unless the organization’s default is private. This value is ignored if the repo already exists.
* **token** (`bool` or `str`, *optional*) —
  The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
  when running `hf auth login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
  is not specified.
* **max\_shard\_size** (`int` or `str`, *optional*, defaults to `"5GB"`) —
  Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
  will then be each of size lower than this size. If expressed as a string, needs to be digits followed
  by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
  Google Colab instances without any CPU OOM issues.
* **create\_pr** (`bool`, *optional*, defaults to `False`) —
  Whether or not to create a PR with the uploaded files or directly commit.
* **safe\_serialization** (`bool`, *optional*, defaults to `True`) —
  Whether or not to convert the model weights in safetensors format for safer serialization.
* **revision** (`str`, *optional*) —
  Branch to push the uploaded files to.
* **commit\_description** (`str`, *optional*) —
  The description of the commit that will be created
* **tags** (`list[str]`, *optional*) —
  List of tags to push on the Hub.

Upload the {object\_files} to the 🤗 Model Hub.

Examples:


```
from transformers import {object_class}

{object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

# Push the {object} to your namespace with the name "my-finetuned-bert".
{object}.push_to_hub("my-finetuned-bert")

# Push the {object} to an organization with the name "my-finetuned-bert".
{object}.push_to_hub("huggingface/my-finetuned-bert")
```

## Sharded checkpoints

#### transformers.modeling\_utils.load\_sharded\_checkpoint

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_utils.py#L387)

( model folder strict = True prefer\_safe = True  ) → `NamedTuple`

Parameters

* **model** (`torch.nn.Module`) — The model in which to load the checkpoint.
* **folder** (`str` or `os.PathLike`) — A path to a folder containing the sharded checkpoint.
* **strict** (`bool`, *optional*, defaults to `True`) —
  Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
* **prefer\_safe** (`bool`, *optional*, defaults to `False`) —
  If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
  safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

Returns

`NamedTuple`

A named tuple with `missing_keys` and `unexpected_keys` fields

* `missing_keys` is a list of str containing the missing keys
* `unexpected_keys` is a list of str containing the unexpected keys

This is the same as
[`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
but for a sharded checkpoint.

This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
loaded in the model.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/model.md)
