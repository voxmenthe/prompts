# Models

The base class [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) implements the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's Hub).

[PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) also implements a few methods which are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary

The other methods that are common to each model are defined in [ModuleUtilsMixin](/docs/transformers/main/en/main_classes/model#transformers.modeling_utils.ModuleUtilsMixin) and [GenerationMixin](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin).

## PreTrainedModel[[transformers.PreTrainedModel]]

#### transformers.PreTrainedModel[[transformers.PreTrainedModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1047)

Base class for all models.

[PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) takes care of storing the configuration of the models and handles methods for loading,
downloading and saving models as well as a few methods common to all models to:

- resize the input embeddings

Class attributes (overridden by derived classes):

- **config_class** ([PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig)) -- A subclass of [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig) to use as configuration class
  for this model architecture.
- **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
  classes of the same architecture adding modules on top of the base model.
- **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
  models, `pixel_values` for vision models and `input_values` for speech models).
- **can_record_outputs** (dict):

push_to_hubtransformers.PreTrainedModel.push_to_hubhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711[{"name": "repo_id", "val": ": str"}, {"name": "commit_message", "val": ": str | None = None"}, {"name": "commit_description", "val": ": str | None = None"}, {"name": "private", "val": ": bool | None = None"}, {"name": "token", "val": ": bool | str | None = None"}, {"name": "revision", "val": ": str | None = None"}, {"name": "create_pr", "val": ": bool = False"}, {"name": "max_shard_size", "val": ": int | str | None = '50GB'"}, {"name": "tags", "val": ": list[str] | None = None"}]- **repo_id** (`str`) --
  The name of the repository you want to push your model to. It should contain your organization name
  when pushing to a given organization.
- **commit_message** (`str`, *optional*) --
  Message to commit while pushing. Will default to `"Upload model"`.
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

Upload the model file to the 🤗 Model Hub.

Examples:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")

# Push the model to your namespace with the name "my-finetuned-bert".
model.push_to_hub("my-finetuned-bert")

# Push the model to an organization with the name "my-finetuned-bert".
model.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your model to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload model"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
#### add_model_tags[[transformers.PreTrainedModel.add_model_tags]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1370)

Add custom tags into the model that gets pushed to the Hugging Face Hub. Will
not overwrite existing tags in the model.

Examples:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")

model.add_model_tags(["custom", "custom-bert"])

# Push the model to your namespace with the name "my-custom-bert".
model.push_to_hub("my-custom-bert")
```

**Parameters:**

tags (`Union[list[str], str]`) : The desired tags to inject in the model
#### can_generate[[transformers.PreTrainedModel.can_generate]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1492)

Returns whether this model can generate sequences with `.generate()` from the `GenerationMixin`.

Under the hood, on classes where this function returns True, some generation-specific changes are triggered:
for instance, the model instance will have a populated `generation_config` attribute.

**Returns:**

``bool``

Whether this model can generate sequences with `.generate()`.
#### dequantize[[transformers.PreTrainedModel.dequantize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1352)

Potentially dequantize the model in case it has been quantized by a quantization method that support
dequantization.
#### disable_input_require_grads[[transformers.PreTrainedModel.disable_input_require_grads]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2024)

Removes the `_require_grads_hook`.
#### enable_input_require_grads[[transformers.PreTrainedModel.enable_input_require_grads]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1991)

Enables the gradients for the input embeddings. This is useful for fine-tuning adapter weights while keeping
the model weights fixed.
#### from_pretrained[[transformers.PreTrainedModel.from_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3585)

Instantiate a pretrained pytorch model from a pre-trained model configuration.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
the model, you should first set it back in training mode with `model.train()`.

The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
task.

The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
weights are discarded.

Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
use this method in a firewalled environment.

Examples:

```python
>>> from transformers import BertConfig, BertModel

>>> # Download model and configuration from huggingface.co and cache.
>>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
>>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
>>> model = BertModel.from_pretrained("./test/saved_model/")
>>> # Update configuration during loading.
>>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
>>> assert model.config.output_attentions == True
```

**Parameters:**

pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*) : Can be either:  - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. - A path to a *directory* containing model weights saved using [save_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `./my_model_directory/`. - `None` if you are both providing the configuration and state dictionary (resp. with keyword arguments `config` and `state_dict`).

model_args (sequence of positional arguments, *optional*) : All remaining positional arguments will be passed to the underlying model's `__init__` method.

config (`Union[PreTrainedConfig, str, os.PathLike]`, *optional*) : Can be either:  - an instance of a class derived from [PreTrainedConfig](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig), - a string or path valid as input to [from_pretrained()](/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.from_pretrained).  Configuration for the model to use instead of an automatically loaded configuration. Configuration can be automatically loaded when:  - The model is a model provided by the library (loaded with the *model id* string of a pretrained model). - The model was saved using [save_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and is reloaded by supplying the save directory. - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a configuration JSON file named *config.json* is found in the directory.

state_dict (`dict[str, torch.Tensor]`, *optional*) : A state dictionary to use instead of a state dictionary loaded from saved weights file.  This option can be used if you want to create a model from a pretrained configuration but load your own weights. In this case though, you should check if using [save_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) and [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) is not a simpler option.

cache_dir (`Union[str, os.PathLike]`, *optional*) : Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.

ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`) : Whether or not to raise an error if some of the weights from the checkpoint do not have the same size as the weights of the model (if for instance, you are instantiating a model with 10 labels from a checkpoint with 3 labels).

force_download (`bool`, *optional*, defaults to `False`) : Whether or not to force the (re-)download of the model weights and configuration files, overriding the cached versions if they exist.

proxies (`dict[str, str]`, *optional*) : A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.

output_loading_info(`bool`, *optional*, defaults to `False`) : Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.

local_files_only(`bool`, *optional*, defaults to `False`) : Whether or not to only look at local files (i.e., do not try to download the model).

token (`str` or `bool`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*, defaults to `"main"`) : The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.    To test a pull request you made on the Hub, you can pass `revision="refs/pr/"`.  

attn_implementation (`str`, *optional*) : The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)), or `"flash_attention_3"` (using [Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.  Accept HF kernel references in the form: /[@][:]  -  and  are any non-"/" and non-":" sequences. - "@" is optional (branch, tag, or commit-ish), e.g. "@main", "@v1.2.0", "@abc123". - ":" is optional and selects a function inside the kernel repo. - Both options can appear together and in this order only: @revision first, then :kernel_name. - We intentionally allow a leading "|" prefix (e.g., "flash|...") because the code strips it before loading; '|' is not excluded in the character classes here.  Examples that match: "org/model" "org/model@main" "org/model:custom_kernel" "org/model@v1.2.3:custom_kernel"
#### get_compiled_call[[transformers.PreTrainedModel.get_compiled_call]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4384)

Return a `torch.compile`'d version of `self.__call__`. This is useful to dynamically choose between
non-compiled/compiled `forward` during inference, especially to switch between prefill (where we don't
want to use compiled version to avoid recomputing the graph with new shapes) and iterative decoding
(where we want the speed-ups of compiled version with static shapes).
#### get_decoder[[transformers.PreTrainedModel.get_decoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2097)

Best-effort lookup of the *decoder* module.

Order of attempts (covers ~85 % of current usages):

1. `self.decoder/self.language_model/self.text_model`
2. `self.base_model`                  (many wrappers store the decoder here)
3. `self.base_model.get_decoder()`    (nested wrappers)
4. fallback: raise for the few exotic models that need a bespoke rule
#### get_encoder[[transformers.PreTrainedModel.get_encoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2039)

Best-effort lookup of the *encoder* module. If provided with `modality` argument,
it looks for a modality-specific encoder in multimodal models (e.g. "image_encoder")
By default the function returns model's text encoder if any, and otherwise returns `self`.

Possible `modality` values are "image", "video" and "audio".
#### get_expanded_tied_weights_keys[[transformers.PreTrainedModel.get_expanded_tied_weights_keys]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2219)

Return the expanded tied weight keys (in case they contain modules or regex patterns) for only the current
model, or recursively for all submodels if `all_submodels=True` (i.e. it will re-check the config values for all
submodels).

For almost all models, we only require to tie the embeddings, so the model has an internal property
`_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}`. In this case, the mapping is already
"expanded", i.e. it already contains full parameters, and this function will simply return a copy of the property.
For more complex patterns, e.g. for `DFineForObjectDetection`, we have the following attribute

```
_tied_weights_keys = {
    r"bbox_embed.(?![0])\d+": "bbox_embed.0",
    r"class_embed.(?![0])\d+": "class_embed.0",
    "model.decoder.class_embed": "class_embed",
    "model.decoder.bbox_embed": "bbox_embed",
}
```

In this case, the function looks up all the model's parameters and buffers, and matches all the params,

returning the following:
```
{
    'bbox_embed.1.layers.0.bias': 'bbox_embed.0.layers.0.bias',
    'bbox_embed.1.layers.0.weight': 'bbox_embed.0.layers.0.weight',
    'bbox_embed.1.layers.1.bias': 'bbox_embed.0.layers.1.bias',
    'bbox_embed.1.layers.1.weight': 'bbox_embed.0.layers.1.weight',
    'bbox_embed.1.layers.2.bias': 'bbox_embed.0.layers.2.bias',
    'bbox_embed.1.layers.2.weight': 'bbox_embed.0.layers.2.weight',
    'bbox_embed.2.layers.0.bias': 'bbox_embed.0.layers.0.bias',
    'bbox_embed.2.layers.0.weight': 'bbox_embed.0.layers.0.weight',
    ...
    'class_embed.1.bias': 'class_embed.0.bias',
    'class_embed.1.weight': 'class_embed.0.weight',
    'class_embed.2.bias': 'class_embed.0.bias',
    'class_embed.2.weight': 'class_embed.0.weight',
    ...
    'model.decoder.class_embed.0.bias': 'class_embed.0.bias',
    'model.decoder.class_embed.0.weight': 'class_embed.0.weight',
    'model.decoder.class_embed.1.bias': 'class_embed.0.bias',
    'model.decoder.class_embed.1.weight': 'class_embed.0.weight',
    ...
    'model.decoder.bbox_embed.0.layers.0.bias': 'bbox_embed.0.layers.0.bias',
    'model.decoder.bbox_embed.0.layers.0.weight': 'bbox_embed.0.layers.0.weight',
    'model.decoder.bbox_embed.0.layers.1.bias': 'bbox_embed.0.layers.1.bias',
    'model.decoder.bbox_embed.0.layers.1.weight': 'bbox_embed.0.layers.1.weight',
    ...
}
```

i.e. all the parameters matching the regex and modules patterns in `_tied_weights_keys`
#### get_memory_footprint[[transformers.PreTrainedModel.get_memory_footprint]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3413)

Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

**Parameters:**

return_buffers (`bool`, *optional*, defaults to `True`) : Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
#### get_parameter_or_buffer[[transformers.PreTrainedModel.get_parameter_or_buffer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4496)

Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
`get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
#### gradient_checkpointing_disable[[transformers.PreTrainedModel.gradient_checkpointing_disable]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2993)

Deactivates gradient checkpointing for the current model.
#### gradient_checkpointing_enable[[transformers.PreTrainedModel.gradient_checkpointing_enable]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2932)

Activates gradient checkpointing for the current model.

We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

**Parameters:**

gradient_checkpointing_kwargs (dict, *optional*) : Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
#### init_weights[[transformers.PreTrainedModel.init_weights]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2921)

Maybe initializes weights. If using a custom `PreTrainedModel`, you need to implement any
initialization logic in `_init_weights`.
#### initialize_weights[[transformers.PreTrainedModel.initialize_weights]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2191)

This is equivalent to calling `self.apply(self._initialize_weights)`, but correctly handles composite models.
This function dynamically dispatches the correct `init_weights` function to the modules as we advance in the
module graph along the recursion. It can handle an arbitrary number of sub-models. Without it, every composite
model would have to recurse a second time on all sub-models explicitly in the outer-most `_init_weights`, which
is extremely error prone and inefficient.
#### mark_tied_weights_as_initialized[[transformers.PreTrainedModel.mark_tied_weights_as_initialized]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4487)

Adds the `_is_hf_initialized` flag on parameters that will be tied, in order to avoid initializing them
later as they will be tied (overwritten) anyway.
This is very important as most embeddings are tied, and they are huge params (vocabularies are often 256k), so
running inits on them is very costly.
#### post_init[[transformers.PreTrainedModel.post_init]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1268)

A method executed at the end of each Transformer model initialization, to execute code that needs the model's
modules properly initialized (such as weight initialization).
#### register_for_auto_class[[transformers.PreTrainedModel.register_for_auto_class]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4247)

Register this class with a given auto class. This should only be used for custom models as the ones in the
library are already mapped with an auto class.

**Parameters:**

auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`) : The auto class to register this new model with.
#### resize_token_embeddings[[transformers.PreTrainedModel.resize_token_embeddings]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2456)

Resizes input token embeddings matrix of the model if `new_num_tokens != config.vocab_size`.

Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

**Parameters:**

new_num_tokens (`int`, *optional*) : The new number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

pad_to_multiple_of (`int`, *optional*) : If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more details about this, or help on choosing the correct value for resizing, refer to this guide: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

mean_resizing (`bool`) : Whether to initialize the added embeddings from a multivariate normal distribution that has old embeddings' mean and covariance or to initialize them with a normal distribution that has a mean of zero and std equals `config.initializer_range`.  Setting `mean_resizing` to `True` is useful when increasing the size of the embeddings of causal language models, where the generated tokens' probabilities won't be affected by the added embeddings because initializing the new embeddings with the old embeddings' mean will reduce the kl-divergence between the next token probability before and after adding the new embeddings. Refer to this article for more information: https://nlp.stanford.edu/~johnhew/vocab-expansion.html

**Returns:**

``torch.nn.Embedding``

Pointer to the input tokens Embeddings Module of the model.
#### save_pretrained[[transformers.PreTrainedModel.save_pretrained]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3020)

Save a model and its configuration file to a directory, so that it can be re-loaded using the
[from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) class method.

**Parameters:**

save_directory (`str` or `os.PathLike`) : Directory to which to save. Will be created if it doesn't exist.

is_main_process (`bool`, *optional*, defaults to `True`) : Whether the process calling this is the main process or not. Useful when in distributed training like TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on the main process to avoid race conditions.

state_dict (nested dictionary of `torch.Tensor`) : The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only save parts of the model or if special precautions need to be taken when recovering the state dictionary of a model (like when using model parallelism).

push_to_hub (`bool`, *optional*, defaults to `False`) : Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).    If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard which will be bigger than `max_shard_size`.   

variant (`str`, *optional*) : If specified, weights are saved in the format model..safetensors.

token (`str` or `bool`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

save_peft_format (`bool`, *optional*, defaults to `True`) : For backward compatibility with PEFT library, in case adapter weights are attached to the model, all keys of the state dict of adapters needs to be prepended with `base_model.model`. Advanced users can disable this behaviours by setting `save_peft_format` to `False`.

save_original_format (`bool`, *optional*, defaults to `True`) : For backward compatibility with the previous versions of `transfomers` you can save the checkpoint with its reverse mapping. The reverse mapping needs to exists even if the model was loaded from a None legacy checkpoint.

kwargs (`dict[str, Any]`, *optional*) : Additional key word arguments passed along to the [push_to_hub()](/docs/transformers/main/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.
#### set_attn_implementation[[transformers.PreTrainedModel.set_attn_implementation]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1892)

Set the requested `attn_implementation` for this model.

**Parameters:**

attn_implementation (`str` or `dict`) : The attention implementation to set for this model. It can be either a `str`, in which case it will be dispatched to all submodels if relevant, or a `dict` where keys are the sub_configs name, in which case each submodel will dispatch the corresponding value.
#### set_decoder[[transformers.PreTrainedModel.set_decoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2120)

Symmetric setter. Mirrors the lookup logic used in `get_decoder`.
#### set_encoder[[transformers.PreTrainedModel.set_encoder]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2071)

Symmetric setter. Mirrors the lookup logic used in `get_encoder`.
#### tie_weights[[transformers.PreTrainedModel.tie_weights]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2330)

Tie the model weights. If `recompute_mapping=False` (default when called internally), it will rely on the
`model.all_tied_weights_keys` attribute, containing the `{target: source}` mapping for the tied params.
If `recompute_mapping=True`, it will re-check all internal submodels and their config to determine the params
that need to be tied. This is the default when `model.tie_weights()` is called on its own, outside of
`__init__`, and `from_pretrained`, in case the config values were changed somewhere.

Note that during `from_pretrained`, tying is *symmetric*: if the mapping says "tie target -> source" but
`source` is missing in the checkpoint while `target` exists, we *swap* source and target so we can still
tie everything to the parameter that actually exists.
#### warn_if_padding_and_no_attention_mask[[transformers.PreTrainedModel.warn_if_padding_and_no_attention_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L4269)

Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.

Custom models should also include a `_supports_assign_param_buffer`, which determines if superfast init can apply
on the particular model. Signs that your model needs this are if `test_save_and_load_from_pretrained` fails. If so,
set this to `False`.

## ModuleUtilsMixin[[transformers.modeling_utils.ModuleUtilsMixin]]

#### transformers.modeling_utils.ModuleUtilsMixin[[transformers.modeling_utils.ModuleUtilsMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L783)

A few utilities for `torch.nn.Modules`, to be used as a mixin.

get_extended_attention_masktransformers.modeling_utils.ModuleUtilsMixin.get_extended_attention_maskhttps://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L856[{"name": "attention_mask", "val": ": Tensor"}, {"name": "input_shape", "val": ": tuple"}, {"name": "device", "val": ": typing.Optional[torch.device] = None"}, {"name": "dtype", "val": ": typing.Optional[torch.dtype] = None"}]- **attention_mask** (`torch.Tensor`) --
  Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
- **input_shape** (`tuple[int]`) --
  The shape of the input to the model.0`torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.

Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

**Parameters:**

attention_mask (`torch.Tensor`) : Mask with ones indicating tokens to attend to, zeros for tokens to ignore.

input_shape (`tuple[int]`) : The shape of the input to the model.

**Returns:**

`torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
#### invert_attention_mask[[transformers.modeling_utils.ModuleUtilsMixin.invert_attention_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L807)

Invert an attention mask (e.g., switches 0. and 1.).

**Parameters:**

encoder_attention_mask (`torch.Tensor`) : An attention mask.

**Returns:**

``torch.Tensor``

The inverted attention mask.
#### num_parameters[[transformers.modeling_utils.ModuleUtilsMixin.num_parameters]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L912)

Get number of (optionally, trainable or non-embeddings) parameters in the module.

**Parameters:**

only_trainable (`bool`, *optional*, defaults to `False`) : Whether or not to return only the number of trainable parameters 

exclude_embeddings (`bool`, *optional*, defaults to `False`) : Whether or not to return only the number of non-embeddings parameters

**Returns:**

``int``

The number of parameters.

## Pushing to the Hub[[transformers.utils.PushToHubMixin]]

#### transformers.utils.PushToHubMixin[[transformers.utils.PushToHubMixin]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L621)

A Mixin containing the functionality to push a model or tokenizer to the hub.

push_to_hubtransformers.utils.PushToHubMixin.push_to_hubhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L711[{"name": "repo_id", "val": ": str"}, {"name": "commit_message", "val": ": str | None = None"}, {"name": "commit_description", "val": ": str | None = None"}, {"name": "private", "val": ": bool | None = None"}, {"name": "token", "val": ": bool | str | None = None"}, {"name": "revision", "val": ": str | None = None"}, {"name": "create_pr", "val": ": bool = False"}, {"name": "max_shard_size", "val": ": int | str | None = '50GB'"}, {"name": "tags", "val": ": list[str] | None = None"}]- **repo_id** (`str`) --
  The name of the repository you want to push your {object} to. It should contain your organization name
  when pushing to a given organization.
- **commit_message** (`str`, *optional*) --
  Message to commit while pushing. Will default to `"Upload {object}"`.
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

Upload the {object_files} to the 🤗 Model Hub.

Examples:

```python
from transformers import {object_class}

{object} = {object_class}.from_pretrained("google-bert/bert-base-cased")

# Push the {object} to your namespace with the name "my-finetuned-bert".
{object}.push_to_hub("my-finetuned-bert")

# Push the {object} to an organization with the name "my-finetuned-bert".
{object}.push_to_hub("huggingface/my-finetuned-bert")
```

**Parameters:**

repo_id (`str`) : The name of the repository you want to push your {object} to. It should contain your organization name when pushing to a given organization.

commit_message (`str`, *optional*) : Message to commit while pushing. Will default to `"Upload {object}"`.

commit_description (`str`, *optional*) : The description of the commit that will be created

private (`bool`, *optional*) : Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.

token (`bool` or `str`, *optional*) : The token to use as HTTP bearer authorization for remote files. If `True` (default), will use the token generated when running `hf auth login` (stored in `~/.huggingface`).

revision (`str`, *optional*) : Branch to push the uploaded files to.

create_pr (`bool`, *optional*, defaults to `False`) : Whether or not to create a PR with the uploaded files or directly commit.

max_shard_size (`int` or `str`, *optional*, defaults to `"50GB"`) : Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).

tags (`list[str]`, *optional*) : List of tags to push on the Hub.
