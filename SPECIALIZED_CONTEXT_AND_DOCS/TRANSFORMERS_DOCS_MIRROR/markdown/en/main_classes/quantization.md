# Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn’t be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren’t supported in Transformers can be added with the `HfQuantizer` class.

Learn how to quantize models in the [Quantization](../quantization) guide.

## QuantoConfig

### class transformers.QuantoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1211)

( weights = 'int8' activations = None modules\_to\_not\_convert: typing.Optional[list] = None \*\*kwargs  )

Parameters

* **weights** (`str`, *optional*, defaults to `"int8"`) —
  The target dtype for the weights after quantization. Supported values are (“float8”,“int8”,“int4”,“int2”)
* **activations** (`str`, *optional*) —
  The target dtype for the activations after quantization. Supported values are (None,“int8”,“float8”)
* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `quanto`.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1239)

( )

Safety checker that arguments are correct

## AqlmConfig

### class transformers.AqlmConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1055)

( in\_group\_size: int = 8 out\_group\_size: int = 1 num\_codebooks: int = 1 nbits\_per\_codebook: int = 16 linear\_weights\_not\_to\_quantize: typing.Optional[list[str]] = None \*\*kwargs  )

Parameters

* **in\_group\_size** (`int`, *optional*, defaults to 8) —
  The group size along the input dimension.
* **out\_group\_size** (`int`, *optional*, defaults to 1) —
  The group size along the output dimension. It’s recommended to always use 1.
* **num\_codebooks** (`int`, *optional*, defaults to 1) —
  Number of codebooks for the Additive Quantization procedure.
* **nbits\_per\_codebook** (`int`, *optional*, defaults to 16) —
  Number of bits encoding a single codebook vector. Codebooks size is 2\*\*nbits\_per\_codebook.
* **linear\_weights\_not\_to\_quantize** (`Optional[list[str]]`, *optional*) —
  List of full paths of `nn.Linear` weight parameters that shall not be quantized.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from which to initialize the configuration object.

This is a wrapper class about `aqlm` parameters.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1092)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

## VptqConfig

### class transformers.VptqConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1170)

( enable\_proxy\_error: bool = False config\_for\_layers: dict = {} shared\_layer\_config: dict = {} modules\_to\_not\_convert: typing.Optional[list] = None \*\*kwargs  )

Parameters

* **enable\_proxy\_error** (`bool`, *optional*, defaults to `False`) — calculate proxy error for each layer
* **config\_for\_layers** (`Dict`, *optional*, defaults to `{}`) — quantization params for each layer
* **shared\_layer\_config** (`Dict`, *optional*, defaults to `{}`) — shared quantization params among layers
* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from which to initialize the configuration object.

This is a wrapper class about `vptq` parameters.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1200)

( )

Safety checker that arguments are correct

## AwqConfig

### class transformers.AwqConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L878)

( bits: int = 4 group\_size: int = 128 zero\_point: bool = True version: AWQLinearVersion = <AWQLinearVersion.GEMM: 'gemm'> backend: AwqBackendPackingMethod = <AwqBackendPackingMethod.AUTOAWQ: 'autoawq'> do\_fuse: typing.Optional[bool] = None fuse\_max\_seq\_len: typing.Optional[int] = None modules\_to\_fuse: typing.Optional[dict] = None modules\_to\_not\_convert: typing.Optional[list] = None exllama\_config: typing.Optional[dict[str, int]] = None \*\*kwargs  )

Parameters

* **bits** (`int`, *optional*, defaults to 4) —
  The number of bits to quantize to.
* **group\_size** (`int`, *optional*, defaults to 128) —
  The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
* **zero\_point** (`bool`, *optional*, defaults to `True`) —
  Whether to use zero point quantization.
* **version** (`AWQLinearVersion`, *optional*, defaults to `AWQLinearVersion.GEMM`) —
  The version of the quantization algorithm to use. GEMM is better for big batch\_size (e.g. >= 8) otherwise,
  GEMV is better (e.g. < 8 ). GEMM models are compatible with Exllama kernels.
* **backend** (`AwqBackendPackingMethod`, *optional*, defaults to `AwqBackendPackingMethod.AUTOAWQ`) —
  The quantization backend. Some models might be quantized using `llm-awq` backend. This is useful for users
  that quantize their own models using `llm-awq` library.
* **do\_fuse** (`bool`, *optional*, defaults to `False`) —
  Whether to fuse attention and mlp layers together for faster inference
* **fuse\_max\_seq\_len** (`int`, *optional*) —
  The Maximum sequence length to generate when using fusing.
* **modules\_to\_fuse** (`dict`, *optional*, default to `None`) —
  Overwrite the natively supported fusing scheme with the one specified by the users.
* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
  Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.
* **exllama\_config** (`dict[str, Any]`, *optional*) —
  You can specify the version of the exllama kernel through the `version` key, the maximum sequence
  length through the `max_input_len` key, and the maximum batch size through the `max_batch_size` key.
  Defaults to `{"version": 2, "max_input_len": 2048, "max_batch_size": 8}` if unset.

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `auto-awq` library awq quantization relying on auto\_awq backend.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L946)

( )

Safety checker that arguments are correct

## EetqConfig

### class transformers.EetqConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1252)

( weights: str = 'int8' modules\_to\_not\_convert: typing.Optional[list] = None \*\*kwargs  )

Parameters

* **weights** (`str`, *optional*, defaults to `"int8"`) —
  The target dtype for the weights. Supported value is only “int8”
* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision.

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `eetq`.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1276)

( )

Safety checker that arguments are correct

## GPTQConfig

### class transformers.GPTQConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L641)

( bits: int tokenizer: typing.Any = None dataset: typing.Union[str, list[str], NoneType] = None group\_size: int = 128 damp\_percent: float = 0.1 desc\_act: bool = False sym: bool = True true\_sequential: bool = True checkpoint\_format: str = 'gptq' meta: typing.Optional[dict[str, typing.Any]] = None backend: typing.Optional[str] = None use\_cuda\_fp16: bool = False model\_seqlen: typing.Optional[int] = None block\_name\_to\_quantize: typing.Optional[str] = None module\_name\_preceding\_first\_block: typing.Optional[list[str]] = None batch\_size: int = 1 pad\_token\_id: typing.Optional[int] = None use\_exllama: typing.Optional[bool] = None max\_input\_length: typing.Optional[int] = None exllama\_config: typing.Optional[dict[str, typing.Any]] = None cache\_block\_outputs: bool = True modules\_in\_block\_to\_quantize: typing.Optional[list[list[str]]] = None \*\*kwargs  )

Parameters

* **bits** (`int`) —
  The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
* **tokenizer** (`str` or `PreTrainedTokenizerBase`, *optional*) —
  The tokenizer used to process the dataset. You can pass either:
  + A custom tokenizer object.
  + A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
  + A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
    using the [save\_pretrained()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained) method, e.g., `./my_model_directory/`.
* **dataset** (`Union[list[str]]`, *optional*) —
  The dataset used for quantization. You can provide your own dataset in a list of string or just use the
  original datasets used in GPTQ paper [‘wikitext2’,‘c4’,‘c4-new’]
* **group\_size** (`int`, *optional*, defaults to 128) —
  The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
* **damp\_percent** (`float`, *optional*, defaults to 0.1) —
  The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
* **desc\_act** (`bool`, *optional*, defaults to `False`) —
  Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
  speed up inference but the perplexity may become slightly worse. Also known as act-order.
* **sym** (`bool`, *optional*, defaults to `True`) —
  Whether to use symmetric quantization.
* **true\_sequential** (`bool`, *optional*, defaults to `True`) —
  Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
  the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
  quantization using inputs that have passed through the previously quantized layers.
* **checkpoint\_format** (`str`, *optional*, defaults to `"gptq"`) —
  GPTQ weight format. `gptq`(v1) is supported by both gptqmodel and auto-gptq. `gptq_v2` is gptqmodel only.
* **meta** (`dict[str, any]`, *optional*) —
  Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta.
  i.e. `meta.quantizer`: [“optimum:*version*”, “gptqmodel:*version*”]
* **backend** (`str`, *optional*) —
  Controls which gptq kernel to be used. Valid values for gptqmodel are `auto`, `auto_trainable` and more. For auto-gptq, only
  valid value is None and `auto_trainable`. Ref gptqmodel backends: <https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py>
* **use\_cuda\_fp16** (`bool`, *optional*, defaults to `False`) —
  Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16. Auto-gptq only.
* **model\_seqlen** (`int`, *optional*) —
  The maximum sequence length that the model can take.
* **block\_name\_to\_quantize** (`str`, *optional*) —
  The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
* **module\_name\_preceding\_first\_block** (`list[str]`, *optional*) —
  The layers that are preceding the first Transformer block.
* **batch\_size** (`int`, *optional*, defaults to 1) —
  The batch size used when processing the dataset
* **pad\_token\_id** (`int`, *optional*) —
  The pad token id. Needed to prepare the dataset when `batch_size` > 1.
* **use\_exllama** (`bool`, *optional*) —
  Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
* **max\_input\_length** (`int`, *optional*) —
  The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
  length. It is specific to the exllama backend with act-order.
* **exllama\_config** (`dict[str, Any]`, *optional*) —
  The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
  to `{"version": 1}` if unset.
* **cache\_block\_outputs** (`bool`, *optional*, defaults to `True`) —
  Whether to cache block outputs to reuse as inputs for the succeeding block.
* **modules\_in\_block\_to\_quantize** (`list[list[str]]`, *optional*) —
  List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
  The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
  Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
  In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
  Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
  better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `optimum` api for gptq quantization relying on auto\_gptq backend.

#### from\_dict\_optimum

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L862)

( config\_dict  )

Get compatible class with optimum gptq config dict

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L773)

( )

Safety checker that arguments are correct

#### to\_dict\_optimum

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L853)

( )

Get compatible dict for optimum gptq config

## BitsAndBytesConfig

### class transformers.BitsAndBytesConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L405)

( load\_in\_8bit = False load\_in\_4bit = False llm\_int8\_threshold = 6.0 llm\_int8\_skip\_modules = None llm\_int8\_enable\_fp32\_cpu\_offload = False llm\_int8\_has\_fp16\_weight = False bnb\_4bit\_compute\_dtype = None bnb\_4bit\_quant\_type = 'fp4' bnb\_4bit\_use\_double\_quant = False bnb\_4bit\_quant\_storage = None \*\*kwargs  )

Parameters

* **load\_in\_8bit** (`bool`, *optional*, defaults to `False`) —
  This flag is used to enable 8-bit quantization with LLM.int8().
* **load\_in\_4bit** (`bool`, *optional*, defaults to `False`) —
  This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from
  `bitsandbytes`.
* **llm\_int8\_threshold** (`float`, *optional*, defaults to 6.0) —
  This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper: <https://huggingface.co/papers/2208.07339> Any hidden states value
  that is above this threshold will be considered an outlier and the operation on those values will be done
  in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but
  there are some exceptional systematic outliers that are very differently distributed for large models.
  These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of
  magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6,
  but a lower threshold might be needed for more unstable models (small models, fine-tuning).
* **llm\_int8\_skip\_modules** (`list[str]`, *optional*) —
  An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as
  Jukebox that has several heads in different places and not necessarily at the last position. For example
  for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.
* **llm\_int8\_enable\_fp32\_cpu\_offload** (`bool`, *optional*, defaults to `False`) —
  This flag is used for advanced use cases and users that are aware of this feature. If you want to split
  your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use
  this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8
  operations will not be run on CPU.
* **llm\_int8\_has\_fp16\_weight** (`bool`, *optional*, defaults to `False`) —
  This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not
  have to be converted back and forth for the backward pass.
* **bnb\_4bit\_compute\_dtype** (`torch.dtype` or str, *optional*, defaults to `torch.float32`) —
  This sets the computational type which might be different than the input type. For example, inputs might be
  fp32, but computation can be set to bf16 for speedups.
* **bnb\_4bit\_quant\_type** (`str`, *optional*, defaults to `"fp4"`) —
  This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types
  which are specified by `fp4` or `nf4`.
* **bnb\_4bit\_use\_double\_quant** (`bool`, *optional*, defaults to `False`) —
  This flag is used for nested quantization where the quantization constants from the first quantization are
  quantized again.
* **bnb\_4bit\_quant\_storage** (`torch.dtype` or str, *optional*, defaults to `torch.uint8`) —
  This sets the storage type to pack the quantized 4-bit params.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from which to initialize the configuration object.

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `bitsandbytes`.

This replaces `load_in_8bit` or `load_in_4bit`therefore both options are mutually exclusive.

Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
then more arguments will be added to this class.

#### is\_quantizable

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L575)

( )

Returns `True` if the model is quantizable, `False` otherwise.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L538)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

#### quantization\_method

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L581)

( )

This method returns the quantization method used for the model. If the model is not quantizable, it returns
`None`.

#### to\_diff\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L612)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

## HfQuantizer

### class transformers.quantizers.HfQuantizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L34)

( quantization\_config: QuantizationConfigMixin \*\*kwargs  )

Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
This class is used only for transformers.PreTrainedModel.from\_pretrained and cannot be easily used outside the scope of that method
yet.

Attributes
quantization\_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
The quantization config that defines the quantization parameters of your model that you want to quantize.
modules\_to\_not\_convert (`list[str]`, *optional*):
The list of module names to not convert when quantizing the model.
required\_packages (`list[str]`, *optional*):
The list of required pip packages to install prior to using the quantizer
requires\_calibration (`bool`):
Whether the quantization method requires to calibrate the model before using it.
requires\_parameters\_quantization (`bool`):
Whether the quantization method requires to create a new Parameter. For example, for bitsandbytes, it is
required to create a new xxxParameter in order to properly quantize the model.

#### adjust\_max\_memory

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L181)

( max\_memory: dict  )

adjust max\_memory argument for infer\_auto\_device\_map() if extra memory is needed for quantization

#### adjust\_target\_dtype

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L109)

( dtype: torch.dtype  )

Parameters

* **dtype** (`torch.dtype`, *optional*) —
  The dtype that is used to compute the device\_map.

Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
to compute the device\_map in case the device\_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

#### check\_quantized\_param

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L185)

( model: PreTrainedModel param\_value: torch.Tensor param\_name: str state\_dict: dict \*\*kwargs  )

checks if a loaded state\_dict component is part of quantized param + some validation; only defined if
requires\_parameters\_quantization == True for quantization methods that require to create a new parameters
for quantization.

#### create\_quantized\_param

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L200)

( \*args \*\*kwargs  )

takes needed components from state\_dict and creates quantized param; only applicable if
requires\_parameters\_quantization == True

#### dequantize

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L267)

( model  )

Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
Note not all quantization schemes support this.

#### get\_accelerator\_warm\_up\_factor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L283)

( )

The factor to be used in `caching_allocator_warmup` to get the number of bytes to pre-allocate to warm up accelerator.
A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means
we allocate half the memory of the weights residing in the empty model, etc…

#### get\_special\_dtypes\_update

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L164)

( model dtype: torch.dtype  )

Parameters

* **model** (`~transformers.PreTrainedModel`) —
  The model to quantize
* **dtype** (`torch.dtype`) —
  The dtype passed in `from_pretrained` method.

returns dtypes for modules that are not quantized - used for the computation of the device\_map in case
one passes a str as a device\_map. The method will use the `modules_to_not_convert` that is modified
in `_process_model_before_weight_loading`.

#### get\_state\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L337)

( model  )

Get state dict. Useful when we need to modify a bit the state dict due to quantization

#### postprocess\_model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L240)

( model: PreTrainedModel \*\*kwargs  )

Parameters

* **model** (`~transformers.PreTrainedModel`) —
  The model to quantize
* **kwargs** (`dict`, *optional*) —
  The keyword arguments that are passed along `_process_model_after_weight_loading`.

Post-process the model post weights loading.
Make sure to override the abstract method `_process_model_after_weight_loading`.

#### preprocess\_model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L222)

( model: PreTrainedModel \*\*kwargs  )

Parameters

* **model** (`~transformers.PreTrainedModel`) —
  The model to quantize
* **kwargs** (`dict`, *optional*) —
  The keyword arguments that are passed along `_process_model_before_weight_loading`.

Setting model attributes and/or converting model before weights loading. At this point
the model should be initialized on the meta device so you can freely manipulate the skeleton
of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

#### remove\_quantization\_config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L253)

( model  )

Remove the quantization config from the model.

#### update\_device\_map

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L97)

( device\_map: typing.Optional[dict[str, typing.Any]]  )

Parameters

* **device\_map** (`Union[dict, str]`, *optional*) —
  The device\_map that is passed through the `from_pretrained` method.

Override this method if you want to pass a override the existing device map with a new
one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device\_map is
passed, the device\_map is set to `“auto”“

#### update\_dtype

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L85)

( dtype: torch.dtype  )

Parameters

* **dtype** (`torch.dtype`) —
  The input dtype that is passed in `from_pretrained`

Some quantization methods require to explicitly set the dtype of the model to a
target dtype. You need to override this method in case you want to make sure that behavior is
preserved

#### update\_expected\_keys

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L152)

( model expected\_keys: list loaded\_keys: list  )

Parameters

* **expected\_keys** (`list[str]`, *optional*) —
  The list of the expected keys in the initialized model.
* **loaded\_keys** (`list[str]`, *optional*) —
  The list of the loaded keys in the checkpoint.

Override this method if you want to adjust the `update_expected_keys`.

#### update\_missing\_keys

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L121)

( model missing\_keys: list prefix: str  )

Parameters

* **missing\_keys** (`list[str]`, *optional*) —
  The list of missing keys in the checkpoint compared to the state dict of the model

Override this method if you want to adjust the `missing_keys`.

#### update\_missing\_keys\_after\_loading

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L141)

( model missing\_keys: list prefix: str  )

Parameters

* **missing\_keys** (`list[str]`, *optional*) —
  The list of missing keys in the checkpoint compared to the state dict of the model

Override this method if you want to adjust the `missing_keys` after loading the model params,
but before the model is post-processed.

#### update\_param\_name

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L299)

( param\_name: str  )

Override this method if you want to adjust the `param_name`.

#### update\_torch\_dtype

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L72)

( dtype: torch.dtype  )

Parameters

* **dtype** (`torch.dtype`) —
  The input dtype that is passed in `from_pretrained`

Deprecared in favor of `update_dtype`!

#### update\_tp\_plan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L218)

( config  )

updates the tp plan for the scales

#### update\_unexpected\_keys

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L131)

( model unexpected\_keys: list prefix: str  )

Parameters

* **unexpected\_keys** (`list[str]`, *optional*) —
  The list of unexpected keys in the checkpoint compared to the state dict of the model

Override this method if you want to adjust the `unexpected_keys`.

#### validate\_environment

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/quantizers/base.py#L210)

( \*args \*\*kwargs  )

This method is used to potentially check for potential conflicts with arguments that are
passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
If no explicit check are needed, simply return nothing.

## HiggsConfig

### class transformers.HiggsConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1499)

( bits: int = 4 p: int = 2 modules\_to\_not\_convert: typing.Optional[list[str]] = None hadamard\_size: int = 512 group\_size: int = 256 tune\_metadata: typing.Optional[dict[str, typing.Any]] = None \*\*kwargs  )

Parameters

* **bits** (int, *optional*, defaults to 4) —
  Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.
* **p** (int, *optional*, defaults to 2) —
  Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.
* **modules\_to\_not\_convert** (`list`, *optional*, default to [“lm\_head”]) —
  List of linear layers that should not be quantized.
* **hadamard\_size** (int, *optional*, defaults to 512) —
  Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.
* **group\_size** (int, *optional*, defaults to 256) —
  Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard\_size.
* **tune\_metadata** (‘dict’, *optional*, defaults to {}) —
  Module-wise metadata (gemm block shapes, GPU metadata, etc.) for saving the kernel tuning results. Default is an empty dictionary. Is set automatically during tuning.

HiggsConfig is a configuration class for quantization using the HIGGS method.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1540)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

## HqqConfig

### class transformers.HqqConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L280)

( nbits: int = 4 group\_size: int = 64 view\_as\_float: bool = False axis: typing.Optional[int] = None dynamic\_config: typing.Optional[dict] = None skip\_modules: list = ['lm\_head'] \*\*kwargs  )

Parameters

* **nbits** (`int`, *optional*, defaults to 4) —
  Number of bits. Supported values are (8, 4, 3, 2, 1).
* **group\_size** (`int`, *optional*, defaults to 64) —
  Group-size value. Supported values are any value that is divisible by weight.shape[axis]).
* **view\_as\_float** (`bool`, *optional*, defaults to `False`) —
  View the quantized weight as float (used in distributed training) if set to `True`.
* **axis** (`Optional[int]`, *optional*) —
  Axis along which grouping is performed. Supported values are 0 or 1.
* **dynamic\_config** (dict, *optional*) —
  Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config.
  If set, each layer specified by its id will use its dedicated quantization configuration.
* **skip\_modules** (`list[str]`, *optional*, defaults to `['lm_head']`) —
  List of `nn.Linear` layers to skip.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from which to initialize the configuration object.

This is wrapper around hqq’s BaseQuantizeConfig.

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L357)

( config: dict  )

Override from\_dict, used in AutoQuantizationConfig.from\_dict in quantizers/auto.py

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L351)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

#### to\_diff\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L382)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

## Mxfp4Config

### class transformers.Mxfp4Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L2055)

( modules\_to\_not\_convert: typing.Optional[list] = None dequantize: bool = False \*\*kwargs  )

Parameters

* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision.
* **dequantize** (`bool`, *optional*, default to `False`) —
  Whether we dequantize the model to bf16 precision or not

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using mxfp4 quantization.

## FbgemmFp8Config

### class transformers.FbgemmFp8Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1468)

( activation\_scale\_ub: float = 1200.0 modules\_to\_not\_convert: typing.Optional[list] = None \*\*kwargs  )

Parameters

* **activation\_scale\_ub** (`float`, *optional*, defaults to 1200.0) —
  The activation scale upper bound. This is used when quantizing the input activation.
* **modules\_to\_not\_convert** (`list`, *optional*, default to `None`) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision.

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using fbgemm fp8 quantization.

## CompressedTensorsConfig

### class transformers.CompressedTensorsConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1285)

( config\_groups: typing.Optional[dict[str, typing.Union[ForwardRef('QuantizationScheme'), list[str]]]] = None format: str = 'dense' quantization\_status: QuantizationStatus = 'initialized' kv\_cache\_scheme: typing.Optional[ForwardRef('QuantizationArgs')] = None global\_compression\_ratio: typing.Optional[float] = None ignore: typing.Optional[list[str]] = None sparsity\_config: typing.Optional[dict[str, typing.Any]] = None quant\_method: str = 'compressed-tensors' run\_compressed: bool = True \*\*kwargs  )

Parameters

* **config\_groups** (`typing.dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.list[str]]]`, *optional*) —
  dictionary mapping group name to a quantization scheme definition
* **format** (`str`, *optional*, defaults to `"dense"`) —
  format the model is represented as. Set `run_compressed` True to execute model as the
  compressed format if not `dense`
* **quantization\_status** (`QuantizationStatus`, *optional*, defaults to `"initialized"`) —
  status of model in the quantization lifecycle, ie ‘initialized’, ‘calibration’, ‘frozen’
* **kv\_cache\_scheme** (`typing.Union[QuantizationArgs, NoneType]`, *optional*) —
  specifies quantization of the kv cache. If None, kv cache is not quantized.
* **global\_compression\_ratio** (`typing.Union[float, NoneType]`, *optional*) —
  0-1 float percentage of model compression
* **ignore** (`typing.Union[typing.list[str], NoneType]`, *optional*) —
  layer names or types to not quantize, supports regex prefixed by ‘re:’
* **sparsity\_config** (`typing.dict[str, typing.Any]`, *optional*) —
  configuration for sparsity compression
* **quant\_method** (`str`, *optional*, defaults to `"compressed-tensors"`) —
  do not override, should be compressed-tensors
* **run\_compressed** (`bool`, *optional*, defaults to `True`) — alter submodules (usually linear) in order to
  emulate compressed model execution if True, otherwise use default submodule

This is a wrapper class that handles compressed-tensors quantization config options.
It is a wrapper around `compressed_tensors.QuantizationConfig`

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1372)

( config\_dict return\_unused\_kwargs = False \*\*kwargs  ) → `QuantizationConfigMixin`

Parameters

* **config\_dict** (`dict[str, Any]`) —
  Dictionary that will be used to instantiate the configuration object.
* **return\_unused\_kwargs** (`bool`,*optional*, defaults to `False`) —
  Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
  `PreTrainedModel`.
* **kwargs** (`dict[str, Any]`) —
  Additional parameters from which to initialize the configuration object.

Returns

`QuantizationConfigMixin`

The configuration object instantiated from those parameters.

Instantiates a [CompressedTensorsConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.CompressedTensorsConfig) from a Python dictionary of parameters.
Optionally unwraps any args from the nested quantization\_config

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1400)

( )

Quantization config to be added to config.json

Serializes this instance to a Python dictionary. Returns:
`dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.

#### to\_diff\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1420)

( ) → `dict[str, Any]`

Returns

`dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance,

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

## TorchAoConfig

### class transformers.TorchAoConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1616)

( quant\_type: typing.Union[str, ForwardRef('AOBaseConfig')] modules\_to\_not\_convert: typing.Optional[list] = None include\_input\_output\_embeddings: bool = False untie\_embedding\_weights: bool = False \*\*kwargs  )

#### from\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1828)

( config\_dict return\_unused\_kwargs = False \*\*kwargs  )

Create configuration from a dictionary.

#### get\_apply\_tensor\_subclass

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1766)

( )

Create the appropriate quantization method based on configuration.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1704)

( )

Validate configuration and set defaults.

#### to\_dict

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1800)

( )

Convert configuration to a dictionary.

## BitNetQuantConfig

### class transformers.BitNetQuantConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1854)

( modules\_to\_not\_convert: typing.Optional[list] = None linear\_class: typing.Optional[str] = 'bitlinear' quantization\_mode: typing.Optional[str] = 'offline' use\_rms\_norm: typing.Optional[bool] = False rms\_norm\_eps: typing.Optional[float] = 1e-06 \*\*kwargs  )

Parameters

* **modules\_to\_not\_convert** (`Optional[List]`, *optional*) —
  Optionally, provides a list of full paths of `nn.Linear` weight parameters
  that shall not be quantized. Defaults to None.
* **linear\_class** (`str`, *optional*, defaults to `"bitlinear"`) —
  The type of linear class to use. Can be either `bitlinear` or `autobitlinear`.
* **quantization\_mode** (`str`, *optional*, defaults to `"offline"`) —
  The quantization mode to use. Can be either `online` or `offline`.
  In `online` mode, the weight quantization parameters are calculated dynamically
  during each forward pass (e.g., based on the current weight values). This can
  adapt to weight changes during training (Quantization-Aware Training - QAT).
  In `offline` mode, quantization parameters are pre-calculated *before* inference.
  These parameters are then fixed and loaded into the quantized model. This
  generally results in lower runtime overhead compared to online quantization.
* **use\_rms\_norm** (`bool`, *optional*, defaults to `False`) —
  Whether to apply RMSNorm on the activations before quantization. This matches the original BitNet paper’s approach
  of normalizing activations before quantization/packing.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon value used in the RMSNorm layer for numerical stability.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments that may be used by specific quantization
  backends or future versions.

Configuration class for applying BitNet quantization.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1903)

( )

Safety checker that arguments are correct

## SpQRConfig

### class transformers.SpQRConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1911)

( bits: int = 3 beta1: int = 16 beta2: int = 16 shapes: typing.Optional[dict[str, int]] = None modules\_to\_not\_convert: typing.Optional[list[str]] = None \*\*kwargs  )

Parameters

* **bits** (`int`, *optional*, defaults to 3) —
  Specifies the bit count for the weights and first order zero-points and scales.
  Currently only bits = 3 is supported.
* **beta1** (`int`, *optional*, defaults to 16) —
  SpQR tile width. Currently only beta1 = 16 is supported.
* **beta2** (`int`, *optional*, defaults to 16) —
  SpQR tile height. Currently only beta2 = 16 is supported.
* **shapes** (`Optional`, *optional*) —
  A dictionary holding the shape of each object. We need this because it’s impossible
  to deduce the exact size of the parameters just from bits, beta1, beta2.
* **modules\_to\_not\_convert** (`Optional[list[str]]`, *optional*) —
  Optionally, provides a list of full paths of `nn.Linear` weight parameters that shall not be quantized.
  Defaults to None.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional parameters from which to initialize the configuration object.

This is a wrapper class about `spqr` parameters. Refer to the original publication for more details.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1952)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

## FineGrainedFP8Config

### class transformers.FineGrainedFP8Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1974)

( activation\_scheme: str = 'dynamic' weight\_block\_size: tuple = (128, 128) modules\_to\_not\_convert: typing.Optional[list] = None \*\*kwargs  )

Parameters

* **activation\_scheme** (`str`, *optional*, defaults to `"dynamic"`) —
  The scheme used for activation, the defaults and only support scheme for now is “dynamic”.
* **weight\_block\_size** (`typing.tuple[int, int]`, *optional*, defaults to `(128, 128)`) —
  The size of the weight blocks for quantization, default is (128, 128).
* **modules\_to\_not\_convert** (`list`, *optional*) —
  A list of module names that should not be converted during quantization.

FineGrainedFP8Config is a configuration class for fine-grained FP8 quantization used mainly for deepseek models.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L2000)

( )

Safety checker that arguments are correct

## QuarkConfig

### class transformers.QuarkConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L2013)

( \*\*kwargs  )

## FPQuantConfig

### class transformers.FPQuantConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1555)

( forward\_dtype: str = 'mxfp4' forward\_method: str = 'abs\_max' backward\_dtype: str = 'bf16' store\_master\_weights: bool = False hadamard\_group\_size: int = 32 pseudoquantization: bool = False modules\_to\_not\_convert: typing.Optional[list[str]] = None \*\*kwargs  )

Parameters

* **forward\_dtype** (`str`, *optional*, defaults to `"mxfp4"`) —
  The dtype to use for the forward pass.
* **forward\_method** (`str`, *optional*, defaults to `"abs_max"`) —
  The scaling to use for the forward pass. Can be `"abs_max"` or `"quest"`. `"abs_max"` is better for PTQ, `"quest"` is better for QAT.
* **backward\_dtype** (`str`, *optional*, defaults to `"bf16"`) —
  The dtype to use for the backward pass.
* **store\_master\_weights** (`bool`, *optional*, defaults to `False`) —
  Whether to store the master weights. Needed for QAT over layer weights.
* **hadamard\_group\_size** (`int`, *optional*, defaults to 32) —
  The group size for the hadamard transform before quantization for `"quest"` it matches the MXFP4 group size (32).
* **pseudoquantization** (`bool`, *optional*, defaults to `False`) —
  Whether to use Triton-based pseudo-quantization. Is mandatory for non-Blackwell GPUs. Doesn’t provide any speedup. For debugging purposes.
* **modules\_to\_not\_convert** (`list`, *optional*) —
  The list of modules to not quantize, useful for quantizing models that explicitly require to have
  some modules left in their original precision.

FPQuantConfig is a configuration class for quantization using the FPQuant method.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L1599)

( )

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

## AutoRoundConfig

### class transformers.AutoRoundConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L211)

( bits: int = 4 group\_size: int = 128 sym: bool = True backend: str = 'auto' \*\*kwargs  )

Parameters

* **bits** (`int`, *optional*, defaults to 4) —
  The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
* **group\_size** (`int`, *optional*, defaults to 128) — Group-size value
* **sym** (`bool`, *optional*, defaults to `True`) — Symmetric quantization or not
* **backend** (`str`, *optional*, defaults to `"auto"`) — The kernel to use, e.g., ipex,marlin, exllamav2, triton, etc. Ref. <https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend>

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded AutoRound quantization.

#### post\_init

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/utils/quantization_config.py#L242)

( )

Safety checker that arguments are correct.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/quantization.md)
