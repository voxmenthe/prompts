# Distributed inference

When a model doesn’t fit on a single GPU, distributed inference with [tensor parallelism](./perf_train_gpu_many#tensor-parallelism) can help. Tensor parallelism shards a model onto multiple accelerators (CUDA GPU, Intel XPU, etc.) and parallelizes computations such as matrix multiplication. It enables fitting larger model sizes into memory and is faster because each accelerator can process a tensor slice.

However, tensor parallelism adds communication overhead and should be used on single machine setups with multiple accelerators to take advantage of fast intra-node communication. For multi-node training, it may be more efficient to use pipeline or data parallelism depending on your use case.

Refer to the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) section on tensor parallelism to learn more.

Check the list below for models that natively support tensor parallelism. Open a GitHub issue or pull request to add support for a model.

Show supported models

* [Cohere](./model_doc/cohere) and [Cohere 2](./model_doc/cohere2)
* [Gemma](./model_doc/gemma) and [Gemma 2](./model_doc/gemma2)
* [GLM](./model_doc/glm)
* [Granite](./model_doc/granite)
* [Llama](./model_doc/llama)
* [Mistral](./model_doc/mistral)
* [Mixtral](./model_doc/mixtral)
* [OLMo](./model_doc/olmo) and [OLMo2](./model_doc/olmo2)
* [Phi](./model_doc/phi) and [Phi-3](./model_doc/phi3)
* [Qwen2](./model_doc/qwen2), [Qwen2Moe](./model_doc/qwen2_moe), and [Qwen2-VL](./model_doc/qwen2_5_vl)
* [Starcoder2](./model_doc/starcoder2)

This guide shows how to enable tensor parallelism with Transformers and different partitioning strategies.

## Partitioning a model

Transformers supports tensor parallelism if a model has a `tp_plan`. There are two plans to partition a model.

* The `auto` tensor parallelism plan partitions a model (see the supported models above) based on a predefined configuration.
* You can also manually specify your own partitioning plan and pass it to the `tp_plan` parameter in [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).

auto plan

manual plan


```
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # better to visualize all the possible strategies
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # better for smaller number of GPUs

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan="auto")
print(model._tp_plan)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

# distributed run
outputs = model(inputs)
```

Launch the inference script above on [torchrun](https://pytorch.org/docs/stable/elastic/run.html) with 4 processes per GPU.


```
torchrun --nproc-per-node 4 demo.py
```

## Partitioning strategies

All partitioning strategies are defined in the `ParallelInterface` class which maps a string to the strategy implementation. You don’t need to interact with this class directly since all the strategies are set with `tp_plan` in [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained), but it is useful for checking what strategies are available.


```
class ParallelInterface(MutableMapping):
    """
    Dict-like object keeping track of allowed attention functions. You can easily add a new attention function
    with a call to `register()`. If a model needs to locally overwrite an existing attention function, say `sdpa`,
    it needs to declare a new instance of this class inside the `modeling_<model>.py`, and declare it on that instance.
    """
    _global_mapping = {
        "colwise": ColwiseParallel(),
        "rowwise": RowwiseParallel(),
        "colwise_rep": ColwiseParallel(output_layouts=Replicate()),
        "rowwise_rep": RowwiseParallel(input_layouts=Replicate()),
        "local_colwise": ColwiseParallel(use_dtensor=False),
        "local_rowwise": RowwiseParallel(use_dtensor=False),
        "local": IsolatedParallel(),
        "gather": GatherParallel(),
        "local_packed_rowwise": PackedRowwiseParallel(use_dtensor=False),
        "sequence_parallel": SequenceParallel(),
        "replicate": ReplicateParallel(),
    }
```

Refer to the table below to learn more about each strategy.

| Strategy | Description |
| --- | --- |
| `ColwiseParallel` | Column-wise partitioning of weights and biases. |
| `RowwiseParallel` | Row-wise partitioning of weights and biases. Also supports partitioning `nn.Embedding` modules. |
| `SequenceParallel` | Sequence parallel implementation to support `LayerNorm` and `Dropout` layers. Also supports Python implementation of [RMSNorm](https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34). |
| `PackedColwiseParallel` | Variant of `ColwiseParallel` to support packed weights (for example, packing `up_proj` and `gate_proj` together). Refer to the [code](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for more details. |
| `PackedRowwiseParallel` | Variant of `RowwiseParallel` to support packed weights (refer to the [code](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for more details). |
| `GatherParallel` | Gather outputs of the module across devices. |
| `IsolatedParallel` | Used for Experts in Mixture-of-Experts (MoE) layers to isolates module from other devices. |
| `ReplicateParallel` | Replicate modules across all devices to prevent `torch.distributed` APIs from breaking due to a partially sharded model. |

### Packed strategies

Weight packing packs multiple linear layers into a single, bigger layer. Packed strategies, `PackedColwiseParallel` and `PackedRowwiseParallel`, are used to shard packed weights. The more basic `ColwiseParallel` or `RowwiseParallel` will incorrectly shard the packed weights.

The example below packs `up_proj` and `gate_proj` into a single `gate_up_proj` module and requires the `PackedRowwiseParallel` strategy to shard `gate_up_proj`.


```
class Llama4TextExperts(nn.Module):
    ...
    self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
```

Batch matrix multiplication can be used in the `forward` pass to compute the output of the `gate_up_proj` module.


```
def forward(self, hidden_states):
    ...
    gate_up = torch.bmm(hidden_states, self.gate_up_proj) # Compute the output of the gate_up_proj module
    gate, up = gate_up.chunk(2, dim=-1) # Split the output into gate and up
```

Refer to [this comment](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py#L79-#L108) for an visual representation of why `Packed*` needs to be used.

### Local strategies

Local strategies (`local_colwise`, `local_rowwise`, `local_packed_rowwise`) don’t use [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) because it isn’t supported for some operations such as [torch.chunk](https://docs.pytorch.org/docs/stable/generated/torch.chunk.html). Instead, local strategies use the basic [torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html) and performs some of the distributed logic manually.

## Custom partitioning strategies

A custom partitioning strategy should inherit from [`TensorParallelLayer`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/tensor_parallel.py) and implement `partition_tensor`, `_prepare_input_fn` and `_prepare_output_fn`.

Then it needs to be registered in the `ParallelInterface` mapping so the dispatching logic can find it when specified in `tp_plan`.

The example below shows how to implement `ColwiseParallel` with this workflow.

1. Inherit from `TensorParallelLayer`. In the `__init__` method, define `input_layouts` and `output_layouts` to describe how the input and output tensors should be placed on devices. The `desired_input_layouts` attribute is used to specify how the input *should* be placed on devices.


   ```
   class ColwiseParallel(TensorParallelLayer):
       def __init__(
           self,
           *,
           input_layouts: Optional[Placement] = None, # The input layout coming from the previous layer
           output_layouts: Optional[Placement] = None, # The output layout we want to achieve
           use_local_output: bool = True, # Whether to use local output or not
           use_dtensor=True, # Whether to use DTensor or not
       ):
           self.input_layouts = (input_layouts or Replicate(),) # The input sharding coming from the previous layer
           self.output_layouts = (output_layouts or Shard(-1),) # Desired output sharding
           self.desired_input_layouts = (Replicate(),) # Desired input sharding, inputs should be replicated across GPUs
           self.use_local_output = use_local_output
           self.use_dtensor = use_dtensor
   ```
2. Implement the `partition_tensor`, `_prepare_input_fn` and `_prepare_output_fn` methods.

   The `partition_tensor` method partitions the tensor and fills `empty_param` with the partitioned tensor. Use the utility function `get_tensor_shard` to help you get the correct shard of the original parameter for a given rank and `get_packed_weights` to help with packed weights.


   ```
   def partition_tensor(
       self,
       param, # Full tensor of the parameter
       empty_param, # Empty tensor of the parameter, will be filled with the partitioned tensor
       param_type, # Type of the parameter, `bias` or `weight`
       param_casting_dtype, # The type to cast the parameter to
       to_contiguous, # Whether to convert the tensor to a contiguous memory layout
       rank, # The rank of the current device
       device_mesh, # The device mesh
   ) -> nn.Parameter: # Return the partitioned parameter
       ...
   ```

   The `_prepare_input_fn` and `_prepare_output_fn` methods are used in the [pre-forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) and [forward](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_hook.html) hooks. They redistribute the inputs and outputs to the desired layout as specified in the `__init__`.


   ```
   def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
       ...
       # Do some custom logic, cast to DTensor etc.
       ...
       return inputs.redistribute(placements=desired_input_layouts, device_mesh=device_mesh)
   def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
       ...
       # Do some custom logic, cast to DTensor etc.
       ...
       return outputs.redistribute(placements=output_layouts, device_mesh=device_mesh)
   ```
3. Register the strategy to `ParallelInterface` to enable it for use with `tp_plan`.


   ```
   from transformers.integrations.tensor_parallel import ParallelInterface

   ParallelInterface.register_strategy("colwise_custom", ColwiseParallel)
   tp_plan = {
       "model.layers.*.self_attn.q_proj": "colwise_custom",
       ...
   }
   model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, tp_plan=tp_plan)
   ```

## Benchmarks

Tensor parallelism can considerably speedup inference, especially for inputs with large batch sizes or long sequences.

Refer to the chart below for the expected speedup for a single forward pass on [Llama](./model_doc/llama) with a sequence length of 512.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Meta-Llama-3-8B-Instruct%2C%20seqlen%20%3D%20512%2C%20python%2C%20w_%20compile.png)

## Design implementation

The Transformers tensor parallelism implementation is framework-agnostic, but for specific implementations, we rely on [DeviceMesh](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) and [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html) from [torch.distributed](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) to provide a simple and extensible interface.

### DeviceMesh

Imagine `DeviceMesh` as a multi-dimensional grid of devices that communicate together. Different parallelization strategies require different types of communication patterns, so you can create a `DeviceMesh` with multiple sub-meshes.


```
from torch.distributed.device_mesh import init_device_mesh

# Create a 1D mesh of 4 GPUs
device_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=["tp"])
```

Most of the `torch.distributed` defined parallelization strategies can be applied to the mesh itself, or its sub-mesh, and it automatically handles the communication patterns.

### DTensor

`DTensor` (Distributed Tensor) is a tensor subclass that handles the distributed logic on top of the usual tensor operations. Most of the model weights in tensor parallelism are stored as `DTensor`s.

The most important part of DTensor is the `placement` attribute because it tells PyTorch how a tensor is placed on the devices in `DeviceMesh`. The `placement` attribute can take the following values.

* `Shard(dimension)` - Indicates how a `DTensor` is sharded across a given dimension, over the `DeviceMesh` it was constructed under. The example below demonstrates how to shard weights over different dimensions for column-wise partitioning.


  ```
  weight = ...
  weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(0)]) # Shard across the 1st (column-wise) dimension
  bias = ...
  bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Shard(-1)]) # Shard across the ONLY dimension
  ```

  This example demonstrates how to shard weights over different dimensions for row-wise partitioning.


  ```
  weight = ...
  weight = DTensor.from_local(weight, device_mesh["tp"], placements=[Shard(1)]) # Shard across the 2nd (row-wise) dimension
  bias = ...
  bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
  ```
* `Replicate()` - Indicates a `DTensor` is replicated across the `DeviceMesh`. It only creates a full copy of the tensor on each device.


  ```
  bias = ...
  bias = DTensor.from_local(bias, device_mesh["tp"], placements=[Replicate()]) # Replicate bias across all GPUs
  ```
* `Partial()` - Indicates a tensor is pending a reduction operation (not typically relevant for usage in Transformers).

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_infer_gpu_multi.md)
