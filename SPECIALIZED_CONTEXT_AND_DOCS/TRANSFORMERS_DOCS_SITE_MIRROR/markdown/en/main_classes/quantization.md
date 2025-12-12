# Quantization

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn't be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes.

Quantization techniques that aren't supported in Transformers can be added with the `HfQuantizer` class.

Learn how to quantize models in the [Quantization](../quantization) guide.

## QuantoConfig[[transformers.QuantoConfig]]

#### transformers.QuantoConfig[[transformers.QuantoConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1018)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `quanto`.

post_inittransformers.QuantoConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1046[]

Safety checker that arguments are correct

**Parameters:**

weights (`str`, *optional*, defaults to `"int8"`) : The target dtype for the weights after quantization. Supported values are ("float8","int8","int4","int2")

activations (`str`, *optional*) : The target dtype for the activations after quantization. Supported values are (None,"int8","float8")

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).

## AqlmConfig[[transformers.AqlmConfig]]

#### transformers.AqlmConfig[[transformers.AqlmConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L862)

This is a wrapper class about `aqlm` parameters.

post_inittransformers.AqlmConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L899[]

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

**Parameters:**

in_group_size (`int`, *optional*, defaults to 8) : The group size along the input dimension.

out_group_size (`int`, *optional*, defaults to 1) : The group size along the output dimension. It's recommended to always use 1.

num_codebooks (`int`, *optional*, defaults to 1) : Number of codebooks for the Additive Quantization procedure.

nbits_per_codebook (`int`, *optional*, defaults to 16) : Number of bits encoding a single codebook vector. Codebooks size is 2**nbits_per_codebook.

linear_weights_not_to_quantize (`Optional[list[str]]`, *optional*) : List of full paths of `nn.Linear` weight parameters that shall not be quantized.

kwargs (`dict[str, Any]`, *optional*) : Additional parameters from which to initialize the configuration object.

## VptqConfig[[transformers.VptqConfig]]

#### transformers.VptqConfig[[transformers.VptqConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L977)

This is a wrapper class about `vptq` parameters.

post_inittransformers.VptqConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1007[]

Safety checker that arguments are correct

**Parameters:**

enable_proxy_error (`bool`, *optional*, defaults to `False`) : calculate proxy error for each layer

config_for_layers (`Dict`, *optional*, defaults to `{}`) : quantization params for each layer

shared_layer_config (`Dict`, *optional*, defaults to `{}`) : shared quantization params among layers

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).

kwargs (`dict[str, Any]`, *optional*) : Additional parameters from which to initialize the configuration object.

## AwqConfig[[transformers.AwqConfig]]

#### transformers.AwqConfig[[transformers.AwqConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L795)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `auto-awq` library awq quantization relying on auto_awq backend.

**Parameters:**

bits (`int`, *optional*, defaults to 4) : The number of bits to quantize to.

group_size (`int`, *optional*, defaults to 128) : The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.

zero_point (`bool`, *optional*, defaults to `True`) : Whether to use zero point quantization.

backend (`AwqBackend`, *optional*, defaults to `AwqBackend.AUTO`) : The quantization backend.

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers). Note you cannot quantize directly with transformers, please refer to `AutoAWQ` documentation for quantizing HF models.

## EetqConfig[[transformers.EetqConfig]]

#### transformers.EetqConfig[[transformers.EetqConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1059)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `eetq`.

post_inittransformers.EetqConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1083[]

Safety checker that arguments are correct

**Parameters:**

weights (`str`, *optional*, defaults to `"int8"`) : The target dtype for the weights. Supported value is only "int8"

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

## GPTQConfig[[transformers.GPTQConfig]]

#### transformers.GPTQConfig[[transformers.GPTQConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L613)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `optimum` api for GPTQ quantization relying on the gptqmodel backend.

from_dict_optimumtransformers.GPTQConfig.from_dict_optimumhttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L784[{"name": "config_dict", "val": ""}]

Get compatible class with optimum gptq config dict

**Parameters:**

bits (`int`) : The number of bits to quantize to, supported numbers are (2, 3, 4, 8).

tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*) : The tokenizer used to process the dataset. You can pass either: - A custom tokenizer object. - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co. - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved using the [save_pretrained()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained) method, e.g., `./my_model_directory/`.

dataset (`Union[list[str]]`, *optional*) : The dataset used for quantization. You can provide your own dataset in a list of string or just use the original datasets used in GPTQ paper ['wikitext2','c4','c4-new']

group_size (`int`, *optional*, defaults to 128) : The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.

damp_percent (`float`, *optional*, defaults to 0.1) : The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.

desc_act (`bool`, *optional*, defaults to `False`) : Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly speed up inference but the perplexity may become slightly worse. Also known as act-order.

act_group_aware (`bool`, *optional*, defaults to `True`) : Use GAR (group aware activation order) during quantization. Has measurable positive impact on quantization quality. Only applicable when `desc_act = False`. Will forced to be `False` when `desc_act = True`.

sym (`bool`, *optional*, defaults to `True`) : Whether to use symmetric quantization.

true_sequential (`bool`, *optional*, defaults to `True`) : Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers.

format (`str`, *optional*, defaults to `"gptq"`) : GPTQ weight format. `gptq` (v1) is supported by gptqmodel. `gptq_v2` is gptqmodel only.

meta (`dict[str, any]`, *optional*) : Properties, such as tooling:version, that do not directly contributes to quantization or quant inference are stored in meta. i.e. `meta.quantizer`: ["optimum:_version_", "gptqmodel:_version_"]

backend (`str`, *optional*) : Controls which kernel to use. Valid values for gptqmodel are `auto`, `auto_trainable` and more. Ref gptqmodel backends: https://github.com/ModelCloud/GPTQModel/blob/main/gptqmodel/utils/backend.py

model_seqlen (`int`, *optional*) : The maximum sequence length that the model can take.

block_name_to_quantize (`str`, *optional*) : The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)

module_name_preceding_first_block (`list[str]`, *optional*) : The layers that are preceding the first Transformer block.

batch_size (`int`, *optional*, defaults to 1) : The batch size used when processing the dataset

pad_token_id (`int`, *optional*) : The pad token id. Needed to prepare the dataset when `batch_size` > 1.

max_input_length (`int`, *optional*) : The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length. It is specific to the exllama backend with act-order.

cache_block_outputs (`bool`, *optional*, defaults to `True`) : Whether to cache block outputs to reuse as inputs for the succeeding block.

modules_in_block_to_quantize (`list[list[str]]`, *optional*) : List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized. The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers. Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`. In this example, we will first quantize the q,k,v layers simultaneously since they are independent. Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
#### post_init[[transformers.GPTQConfig.post_init]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L734)

Safety checker that arguments are correct
#### to_dict_optimum[[transformers.GPTQConfig.to_dict_optimum]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L778)

Get compatible dict for optimum gptq config

## BitsAndBytesConfig[[transformers.BitsAndBytesConfig]]

#### transformers.BitsAndBytesConfig[[transformers.BitsAndBytesConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L386)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using `bitsandbytes`.

Currently only supports `LLM.int8()`, `FP4`, and `NF4` quantization. If more methods are added to `bitsandbytes`,
then more arguments will be added to this class.

is_quantizabletransformers.BitsAndBytesConfig.is_quantizablehttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L547[]

Returns `True` if the model is quantizable, `False` otherwise.

**Parameters:**

load_in_8bit (`bool`, *optional*, defaults to `False`) : This flag is used to enable 8-bit quantization with LLM.int8().

load_in_4bit (`bool`, *optional*, defaults to `False`) : This flag is used to enable 4-bit quantization by replacing the Linear layers with FP4/NF4 layers from `bitsandbytes`.

llm_int8_threshold (`float`, *optional*, defaults to 6.0) : This corresponds to the outlier threshold for outlier detection as described in `LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale` paper: https://huggingface.co/papers/2208.07339 Any hidden states value that is above this threshold will be considered an outlier and the operation on those values will be done in fp16. Values are usually normally distributed, that is, most values are in the range [-3.5, 3.5], but there are some exceptional systematic outliers that are very differently distributed for large models. These outliers are often in the interval [-60, -6] or [6, 60]. Int8 quantization works well for values of magnitude ~5, but beyond that, there is a significant performance penalty. A good default threshold is 6, but a lower threshold might be needed for more unstable models (small models, fine-tuning).

llm_int8_skip_modules (`list[str]`, *optional*) : An explicit list of the modules that we do not want to convert in 8-bit. This is useful for models such as Jukebox that has several heads in different places and not necessarily at the last position. For example for `CausalLM` models, the last `lm_head` is kept in its original `dtype`.

llm_int8_enable_fp32_cpu_offload (`bool`, *optional*, defaults to `False`) : This flag is used for advanced use cases and users that are aware of this feature. If you want to split your model in different parts and run some parts in int8 on GPU and some parts in fp32 on CPU, you can use this flag. This is useful for offloading large models such as `google/flan-t5-xxl`. Note that the int8 operations will not be run on CPU.

llm_int8_has_fp16_weight (`bool`, *optional*, defaults to `False`) : This flag runs LLM.int8() with 16-bit main weights. This is useful for fine-tuning as the weights do not have to be converted back and forth for the backward pass.

bnb_4bit_compute_dtype (`torch.dtype` or str, *optional*, defaults to `torch.float32`) : This sets the computational type which might be different than the input type. For example, inputs might be fp32, but computation can be set to bf16 for speedups.

bnb_4bit_quant_type (`str`,  *optional*, defaults to `"fp4"`) : This sets the quantization data type in the bnb.nn.Linear4Bit layers. Options are FP4 and NF4 data types which are specified by `fp4` or `nf4`.

bnb_4bit_use_double_quant (`bool`, *optional*, defaults to `False`) : This flag is used for nested quantization where the quantization constants from the first quantization are quantized again.

bnb_4bit_quant_storage (`torch.dtype` or str, *optional*, defaults to `torch.uint8`) : This sets the storage type to pack the quantized 4-bit params.

kwargs (`dict[str, Any]`, *optional*) : Additional parameters from which to initialize the configuration object.
#### post_init[[transformers.BitsAndBytesConfig.post_init]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L517)

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
#### quantization_method[[transformers.BitsAndBytesConfig.quantization_method]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L553)

This method returns the quantization method used for the model. If the model is not quantizable, it returns
`None`.
#### to_diff_dict[[transformers.BitsAndBytesConfig.to_diff_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L584)

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this configuration instance,

## HfQuantizer[[transformers.quantizers.HfQuantizer]]

#### transformers.quantizers.HfQuantizer[[transformers.quantizers.HfQuantizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L71)

Abstract class of the HuggingFace quantizer. Supports for now quantizing HF transformers models for inference and/or quantization.
This class is used only for transformers.PreTrainedModel.from_pretrained and cannot be easily used outside the scope of that method
yet.

Attributes
quantization_config (`transformers.utils.quantization_config.QuantizationConfigMixin`):
The quantization config that defines the quantization parameters of your model that you want to quantize.
requires_calibration (`bool`):
Whether the quantization method requires to calibrate the model before using it.

adjust_max_memorytransformers.quantizers.HfQuantizer.adjust_max_memoryhttps://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L150[{"name": "max_memory", "val": ": dict"}]
adjust max_memory argument for infer_auto_device_map() if extra memory is needed for quantization
#### adjust_target_dtype[[transformers.quantizers.HfQuantizer.adjust_target_dtype]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L121)

Override this method if you want to adjust the `target_dtype` variable used in `from_pretrained`
to compute the device_map in case the device_map is a `str`. E.g. for bitsandbytes we force-set `target_dtype`
to `torch.int8` and for 4-bit we pass a custom enum `accelerate.CustomDtype.int4`.

**Parameters:**

dtype (`torch.dtype`, *optional*) : The dtype that is used to compute the device_map.
#### dequantize[[transformers.quantizers.HfQuantizer.dequantize]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L240)

Potentially dequantize the model to retrieve the original model, with some loss in accuracy / performance.
Note not all quantization schemes support this.
#### get_accelerator_warm_up_factor[[transformers.quantizers.HfQuantizer.get_accelerator_warm_up_factor]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L250)

The factor to be used in `caching_allocator_warmup` to get the number of bytes to pre-allocate to warm up accelerator.
A factor of 2 means we allocate all bytes in the empty model (since we allocate in fp16), a factor of 4 means
we allocate half the memory of the weights residing in the empty model, etc...
#### get_param_name[[transformers.quantizers.HfQuantizer.get_param_name]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L266)

Override this method if you want to adjust the `param_name`.
#### get_state_dict_and_metadata[[transformers.quantizers.HfQuantizer.get_state_dict_and_metadata]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L304)

Get state dict and metadata. Useful when we need to modify a bit the state dict due to quantization
#### param_element_size[[transformers.quantizers.HfQuantizer.param_element_size]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L133)

Return the element size (in bytes) for `param_name`.
#### param_needs_quantization[[transformers.quantizers.HfQuantizer.param_needs_quantization]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L154)

Check whether a given param needs to be quantized.
#### postprocess_model[[transformers.quantizers.HfQuantizer.postprocess_model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L208)

Post-process the model post weights loading.
Make sure to override the abstract method `_process_model_after_weight_loading`.

**Parameters:**

model (`~transformers.PreTrainedModel`) : The model to quantize

kwargs (`dict`, *optional*) : The keyword arguments that are passed along `_process_model_after_weight_loading`.
#### preprocess_model[[transformers.quantizers.HfQuantizer.preprocess_model]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L179)

Setting model attributes and/or converting model before weights loading. At this point
the model should be initialized on the meta device so you can freely manipulate the skeleton
of the model in order to replace modules in-place. Make sure to override the abstract method `_process_model_before_weight_loading`.

**Parameters:**

model (`~transformers.PreTrainedModel`) : The model to quantize

kwargs (`dict`, *optional*) : The keyword arguments that are passed along `_process_model_before_weight_loading`.
#### remove_quantization_config[[transformers.quantizers.HfQuantizer.remove_quantization_config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L226)

Remove the quantization config from the model.
#### update_device_map[[transformers.quantizers.HfQuantizer.update_device_map]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L109)

Override this method if you want to pass a override the existing device map with a new
one. E.g. for bitsandbytes, since `accelerate` is a hard requirement, if no device_map is
passed, the device_map is set to `"auto"``

**Parameters:**

device_map (`Union[dict, str]`, *optional*) : The device_map that is passed through the `from_pretrained` method.
#### update_dtype[[transformers.quantizers.HfQuantizer.update_dtype]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L97)

Some quantization methods require to explicitly set the dtype of the model to a
target dtype. You need to override this method in case you want to make sure that behavior is
preserved

**Parameters:**

dtype (`torch.dtype`) : The input dtype that is passed in `from_pretrained`
#### update_ep_plan[[transformers.quantizers.HfQuantizer.update_ep_plan]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L172)

updates the tp plan for the scales
#### update_tp_plan[[transformers.quantizers.HfQuantizer.update_tp_plan]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L168)

updates the tp plan for the scales
#### validate_environment[[transformers.quantizers.HfQuantizer.validate_environment]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/base.py#L160)

This method is used to potentially check for potential conflicts with arguments that are
passed in `from_pretrained`. You need to define it for all future quantizers that are integrated with transformers.
If no explicit check are needed, simply return nothing.

## HiggsConfig[[transformers.HiggsConfig]]

#### transformers.HiggsConfig[[transformers.HiggsConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1306)

HiggsConfig is a configuration class for quantization using the HIGGS method.

post_inittransformers.HiggsConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1347[]

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

**Parameters:**

bits (int, *optional*, defaults to 4) : Number of bits to use for quantization. Can be 2, 3 or 4. Default is 4.

p (int, *optional*, defaults to 2) : Quantization grid dimension. 1 and 2 are supported. 2 is always better in practice. Default is 2.

modules_to_not_convert (`list`, *optional*, default to ["lm_head"]) : List of linear layers that should not be quantized.

hadamard_size (int, *optional*, defaults to 512) : Hadamard size for the HIGGS method. Default is 512. Input dimension of matrices is padded to this value. Decreasing this below 512 will reduce the quality of the quantization.

group_size (int, *optional*, defaults to 256) : Group size for the HIGGS method. Can be 64, 128 or 256. Decreasing it barely affects the performance. Default is 256. Must be a divisor of hadamard_size.

tune_metadata ('dict', *optional*, defaults to {}) : Module-wise metadata (gemm block shapes, GPU metadata, etc.) for saving the kernel tuning results. Default is an empty dictionary. Is set automatically during tuning.

## HqqConfig[[transformers.HqqConfig]]

#### transformers.HqqConfig[[transformers.HqqConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L273)

This is wrapper around hqq's BaseQuantizeConfig.

from_dicttransformers.HqqConfig.from_dicthttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L338[{"name": "config", "val": ": dict"}]

Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py

**Parameters:**

nbits (`int`, *optional*, defaults to 4) : Number of bits. Supported values are (8, 4, 3, 2, 1).

group_size (`int`, *optional*, defaults to 64) : Group-size value. Supported values are any value that is divisible by weight.shape[axis]).

view_as_float (`bool`, *optional*, defaults to `False`) : View the quantized weight as float (used in distributed training) if set to `True`.

axis (`Optional[int]`, *optional*) : Axis along which grouping is performed. Supported values are 0 or 1.

dynamic_config (dict, *optional*) : Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config. If set, each layer specified by its id will use its dedicated quantization configuration.

skip_modules (`list[str]`, *optional*, defaults to `['lm_head']`) : List of `nn.Linear` layers to skip.

kwargs (`dict[str, Any]`, *optional*) : Additional parameters from which to initialize the configuration object.
#### post_init[[transformers.HqqConfig.post_init]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L333)

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
#### to_diff_dict[[transformers.HqqConfig.to_diff_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L363)

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this configuration instance,

## Mxfp4Config[[transformers.Mxfp4Config]]

#### transformers.Mxfp4Config[[transformers.Mxfp4Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1896)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using mxfp4 quantization.

**Parameters:**

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

dequantize (`bool`, *optional*, default to `False`) : Whether we dequantize the model to bf16 precision or not

## FbgemmFp8Config[[transformers.FbgemmFp8Config]]

#### transformers.FbgemmFp8Config[[transformers.FbgemmFp8Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1275)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded using fbgemm fp8 quantization.

**Parameters:**

activation_scale_ub (`float`, *optional*, defaults to 1200.0) : The activation scale upper bound. This is used when quantizing the input activation.

modules_to_not_convert (`list`, *optional*, default to `None`) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

## CompressedTensorsConfig[[transformers.CompressedTensorsConfig]]

#### transformers.CompressedTensorsConfig[[transformers.CompressedTensorsConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1092)

This is a wrapper class that handles compressed-tensors quantization config options.
It is a wrapper around `compressed_tensors.QuantizationConfig`

from_dicttransformers.CompressedTensorsConfig.from_dicthttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1179[{"name": "config_dict", "val": ""}, {"name": "return_unused_kwargs", "val": " = False"}, {"name": "**kwargs", "val": ""}]- **config_dict** (`dict[str, Any]`) --
  Dictionary that will be used to instantiate the configuration object.
- **return_unused_kwargs** (`bool`,*optional*, defaults to `False`) --
  Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
  `PreTrainedModel`.
- **kwargs** (`dict[str, Any]`) --
  Additional parameters from which to initialize the configuration object.0`QuantizationConfigMixin`The configuration object instantiated from those parameters.

Instantiates a [CompressedTensorsConfig](/docs/transformers/main/en/main_classes/quantization#transformers.CompressedTensorsConfig) from a Python dictionary of parameters.
Optionally unwraps any args from the nested quantization_config

**Parameters:**

config_groups (`typing.dict[str, typing.Union[ForwardRef('QuantizationScheme'), typing.list[str]]]`, *optional*) : dictionary mapping group name to a quantization scheme definition

format (`str`, *optional*, defaults to `"dense"`) : format the model is represented as. Set `run_compressed` True to execute model as the compressed format if not `dense`

quantization_status (`QuantizationStatus`, *optional*, defaults to `"initialized"`) : status of model in the quantization lifecycle, ie 'initialized', 'calibration', 'frozen'

kv_cache_scheme (`typing.Union[QuantizationArgs, NoneType]`, *optional*) : specifies quantization of the kv cache. If None, kv cache is not quantized.

global_compression_ratio (`typing.Union[float, NoneType]`, *optional*) : 0-1 float percentage of model compression

ignore (`typing.Union[typing.list[str], NoneType]`, *optional*) : layer names or types to not quantize, supports regex prefixed by 're:'

sparsity_config (`typing.dict[str, typing.Any]`, *optional*) : configuration for sparsity compression

quant_method (`str`, *optional*, defaults to `"compressed-tensors"`) : do not override, should be compressed-tensors

run_compressed (`bool`, *optional*, defaults to `True`) : alter submodules (usually linear) in order to emulate compressed model execution if True, otherwise use default submodule

**Returns:**

``QuantizationConfigMixin``

The configuration object instantiated from those parameters.
#### to_dict[[transformers.CompressedTensorsConfig.to_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1207)

Quantization config to be added to config.json

Serializes this instance to a Python dictionary. Returns:
`dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
#### to_diff_dict[[transformers.CompressedTensorsConfig.to_diff_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1227)

Removes all attributes from config which correspond to the default config attributes for better readability and
serializes to a Python dictionary.

**Returns:**

``dict[str, Any]``

Dictionary of all the attributes that make up this configuration instance,

## TorchAoConfig[[transformers.TorchAoConfig]]

#### transformers.TorchAoConfig[[transformers.TorchAoConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1451)

from_dicttransformers.TorchAoConfig.from_dicthttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1663[{"name": "config_dict", "val": ""}, {"name": "return_unused_kwargs", "val": " = False"}, {"name": "**kwargs", "val": ""}]
Create configuration from a dictionary.
#### get_apply_tensor_subclass[[transformers.TorchAoConfig.get_apply_tensor_subclass]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1601)

Create the appropriate quantization method based on configuration.
#### post_init[[transformers.TorchAoConfig.post_init]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1539)

Validate configuration and set defaults.
#### to_dict[[transformers.TorchAoConfig.to_dict]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1635)

Convert configuration to a dictionary.

## BitNetQuantConfig[[transformers.BitNetQuantConfig]]

#### transformers.BitNetQuantConfig[[transformers.BitNetQuantConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1689)

Configuration class for applying BitNet quantization.

post_inittransformers.BitNetQuantConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1738[]

Safety checker that arguments are correct

**Parameters:**

modules_to_not_convert (`Optional[List]`, *optional*) : Optionally, provides a list of full paths of `nn.Linear` weight parameters that shall not be quantized. Defaults to None.

linear_class (`str`, *optional*, defaults to `"bitlinear"`) : The type of linear class to use. Can be either `bitlinear` or `autobitlinear`.

quantization_mode (`str`, *optional*, defaults to `"offline"`) : The quantization mode to use. Can be either `online` or `offline`. In `online` mode, the weight quantization parameters are calculated dynamically during each forward pass (e.g., based on the current weight values). This can adapt to weight changes during training (Quantization-Aware Training - QAT). In `offline` mode, quantization parameters are pre-calculated *before* inference. These parameters are then fixed and loaded into the quantized model. This generally results in lower runtime overhead compared to online quantization.

use_rms_norm (`bool`, *optional*, defaults to `False`) : Whether to apply RMSNorm on the activations before quantization. This matches the original BitNet paper's approach of normalizing activations before quantization/packing.

rms_norm_eps (`float`, *optional*, defaults to 1e-06) : The epsilon value used in the RMSNorm layer for numerical stability.

kwargs (`dict[str, Any]`, *optional*) : Additional keyword arguments that may be used by specific quantization backends or future versions.

## SpQRConfig[[transformers.SpQRConfig]]

#### transformers.SpQRConfig[[transformers.SpQRConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1745)

This is a wrapper class about `spqr` parameters. Refer to the original publication for more details.

post_inittransformers.SpQRConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1786[]

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

**Parameters:**

bits (`int`, *optional*, defaults to 3) : Specifies the bit count for the weights and first order zero-points and scales. Currently only bits = 3 is supported.

beta1 (`int`, *optional*, defaults to 16) : SpQR tile width. Currently only beta1 = 16 is supported.

beta2 (`int`, *optional*, defaults to 16) : SpQR tile height. Currently only beta2 = 16 is supported.

shapes (`Optional`, *optional*) : A dictionary holding the shape of each object. We need this because it's impossible to deduce the exact size of the parameters just from bits, beta1, beta2.

modules_to_not_convert (`Optional[list[str]]`, *optional*) : Optionally, provides a list of full paths of `nn.Linear` weight parameters that shall not be quantized. Defaults to None.

kwargs (`dict[str, Any]`, *optional*) : Additional parameters from which to initialize the configuration object.

## FineGrainedFP8Config[[transformers.FineGrainedFP8Config]]

#### transformers.FineGrainedFP8Config[[transformers.FineGrainedFP8Config]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1808)

FineGrainedFP8Config is a configuration class for fine-grained FP8 quantization used mainly for deepseek models.

post_inittransformers.FineGrainedFP8Config.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1838[]

Safety checker that arguments are correct

**Parameters:**

activation_scheme (`str`, *optional*, defaults to `"dynamic"`) : The scheme used for activation, the defaults and only support scheme for now is "dynamic".

weight_block_size (`typing.tuple[int, int]`, *optional*, defaults to `(128, 128)`) : The size of the weight blocks for quantization, default is (128, 128).

dequantize (`bool`, *optional*, defaults to `False`) : Whether to dequantize the model during loading.

modules_to_not_convert (`list`, *optional*) : A list of module names that should not be converted during quantization.

## QuarkConfig[[transformers.QuarkConfig]]

#### transformers.QuarkConfig[[transformers.QuarkConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1854)

## FPQuantConfig[[transformers.FPQuantConfig]]

#### transformers.FPQuantConfig[[transformers.FPQuantConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1362)

FPQuantConfig is a configuration class for quantization using the FPQuant method.

post_inittransformers.FPQuantConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L1409[]

Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.

**Parameters:**

forward_dtype (`str`, *optional*, defaults to `"nvfp4"`) : The dtype to use for the forward pass.

forward_method (`str`, *optional*, defaults to `"abs_max"`) : The scaling to use for the forward pass. Can be `"abs_max"` or `"quest"`. `"abs_max"` is better for PTQ, `"quest"` is better for QAT.

backward_dtype (`str`, *optional*, defaults to `"bf16"`) : The dtype to use for the backward pass.

store_master_weights (`bool`, *optional*, defaults to `False`) : Whether to store the master weights. Needed for QAT over layer weights.

hadamard_group_size (`int`, *optional*) : The group size for the hadamard transform before quantization for `"quest"` it matches the MXFP4 group size (32). If `None`, it will be set to 16 for `"nvfp4"` and 32 for `"mxfp4"`.

pseudoquantization (`bool`, *optional*, defaults to `False`) : Whether to use Triton-based pseudo-quantization. Is mandatory for non-Blackwell GPUs. Doesn't provide any speedup. For debugging purposes.

transform_init (`str`, *optional*, defaults to `"hadamard"`) : a method to initialize the pre-processing matrix with. Can be `"hadamard"`, `"identity"` or `"gsr"`.

modules_to_not_convert (`list`, *optional*) : The list of modules to not quantize, useful for quantizing models that explicitly require to have some modules left in their original precision.

## AutoRoundConfig[[transformers.AutoRoundConfig]]

#### transformers.AutoRoundConfig[[transformers.AutoRoundConfig]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L204)

This is a wrapper class about all possible attributes and features that you can play with a model that has been
loaded AutoRound quantization.

post_inittransformers.AutoRoundConfig.post_inithttps://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L235[]
Safety checker that arguments are correct.

**Parameters:**

bits (`int`, *optional*, defaults to 4) : The number of bits to quantize to, supported numbers are (2, 3, 4, 8).

group_size (`int`, *optional*, defaults to 128) : Group-size value

sym (`bool`, *optional*, defaults to `True`) : Symmetric quantization or not

backend (`str`, *optional*, defaults to `"auto"`) : The kernel to use, e.g., ipex,marlin, exllamav2, triton, etc. Ref. https://github.com/intel/auto-round?tab=readme-ov-file#specify-backend
