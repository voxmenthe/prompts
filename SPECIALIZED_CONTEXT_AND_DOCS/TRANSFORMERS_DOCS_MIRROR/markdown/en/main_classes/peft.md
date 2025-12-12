# PEFT

The [PeftAdapterMixin](/docs/transformers/v4.56.2/en/main_classes/peft#transformers.integrations.PeftAdapterMixin) provides functions from the [PEFT](https://huggingface.co/docs/peft/index) library for managing adapters with Transformers. This mixin currently supports LoRA, IA3, and AdaLora. Prefix tuning methods (prompt tuning, prompt learning) aren’t supported because they can’t be injected into a torch module.

### class transformers.integrations.PeftAdapterMixin

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L67)

( )

A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT
library: <https://huggingface.co/docs/peft/index>

Currently supported PEFT methods are all non-prefix tuning methods. Below is the list of supported PEFT methods
that anyone can load, train and run with this mixin class:

* Low Rank Adapters (LoRA): <https://huggingface.co/docs/peft/conceptual_guides/lora>
* IA3: <https://huggingface.co/docs/peft/conceptual_guides/ia3>
* AdaLora: <https://huggingface.co/papers/2303.10512>

Other PEFT models such as prompt tuning, prompt learning are out of scope as these adapters are not “injectable”
into a torch module. For using these methods, please refer to the usage guide of PEFT library.

With this mixin, if the correct PEFT version is installed, it is possible to:

* Load an adapter stored on a local path or in a remote Hub repository, and inject it in the model
* Attach new adapters in the model and train them with Trainer or by your own.
* Attach multiple adapters and iteratively activate / deactivate them
* Activate / deactivate all adapters from the model.
* Get the `state_dict` of the active adapter.

#### load\_adapter

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L93)

( peft\_model\_id: typing.Optional[str] = None adapter\_name: typing.Optional[str] = None revision: typing.Optional[str] = None token: typing.Optional[str] = None device\_map: typing.Optional[str] = 'auto' max\_memory: typing.Optional[str] = None offload\_folder: typing.Optional[str] = None offload\_index: typing.Optional[int] = None peft\_config: typing.Optional[dict[str, typing.Any]] = None adapter\_state\_dict: typing.Optional[dict[str, 'torch.Tensor']] = None low\_cpu\_mem\_usage: bool = False is\_trainable: bool = False adapter\_kwargs: typing.Optional[dict[str, typing.Any]] = None  )

Parameters

* **peft\_model\_id** (`str`, *optional*) —
  The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
  and adapter weights.
* **adapter\_name** (`str`, *optional*) —
  The adapter name to use. If not set, will use the default adapter.
* **revision** (`str`, *optional*, defaults to `"main"`) —
  The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
  git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
  identifier allowed by git.

  To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.
* **token** (`str`, `optional`) —
  Whether to use authentication token to load the remote folder. Useful to load private repositories
  that are on HuggingFace Hub. You might need to call `hf auth login` and paste your tokens to
  cache it.
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
  A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
  GPU and the available CPU RAM if unset.
* **offload\_folder** (`str` or `os.PathLike`, `optional`) —
  If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
* **offload\_index** (`int`, `optional`) —
  `offload_index` argument to be passed to `accelerate.dispatch_model` method.
* **peft\_config** (`dict[str, Any]`, *optional*) —
  The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
  methods. This argument is used in case users directly pass PEFT state dicts
* **adapter\_state\_dict** (`dict[str, torch.Tensor]`, *optional*) —
  The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
  dicts
* **low\_cpu\_mem\_usage** (`bool`, *optional*, defaults to `False`) —
  Reduce memory usage while loading the PEFT adapter. This should also speed up the loading process.
  Requires PEFT version 0.13.0 or higher.
* **is\_trainable** (`bool`, *optional*, defaults to `False`) —
  Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
  used for inference.
* **adapter\_kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
  `find_adapter_config_file` method.

Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
invite you to read more about them on PEFT official documentation: <https://huggingface.co/docs/peft>

Requires peft as a backend to load the adapter weights.

#### add\_adapter

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L314)

( adapter\_config adapter\_name: typing.Optional[str] = None  )

Parameters

* **adapter\_config** (`~peft.PeftConfig`) —
  The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
  methods
* **adapter\_name** (`str`, *optional*, defaults to `"default"`) —
  The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use “default” as the
default adapter name).

#### set\_adapter

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L351)

( adapter\_name: typing.Union[list[str], str]  )

Parameters

* **adapter\_name** (`Union[list[str], str]`) —
  The name of the adapter to set. Can be also a list of strings to set multiple adapters.

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

#### disable\_adapters

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L396)

( )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Disable all adapters that are attached to the model. This leads to inferring with the base model only.

#### enable\_adapters

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L419)

( )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Enable adapters that are attached to the model.

#### active\_adapters

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L441)

( )

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
for inference) returns the list of all active adapters so that users can deal with them accordingly.

For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
a single string.

#### get\_adapter\_state\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/integrations/peft.py#L480)

( adapter\_name: typing.Optional[str] = None state\_dict: typing.Optional[dict] = None  )

Parameters

* **adapter\_name** (`str`, *optional*) —
  The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
* **state\_dict** (nested dictionary of `torch.Tensor`, *optional*) —
  The state dictionary of the model. Will default to `self.state_dict()`, but can be used if special
  precautions need to be taken when recovering the state dictionary of a model (like when using model
  parallelism).

If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
official documentation: <https://huggingface.co/docs/peft>

Gets the adapter state dict that should only contain the weights tensors of the specified adapter\_name adapter.
If no adapter\_name is passed, the active adapter is used.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/peft.md)
