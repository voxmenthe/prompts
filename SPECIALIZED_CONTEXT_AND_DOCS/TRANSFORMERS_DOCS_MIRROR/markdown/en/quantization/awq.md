# AWQ

[Activation-aware Weight Quantization (AWQ)](https://hf.co/papers/2306.00978) preserves a small fraction of the weights that are important for LLM performance to compress a model to 4-bits with minimal performance degradation.

There are several libraries for quantizing models with the AWQ algorithm, such as [llm-awq](https://github.com/mit-han-lab/llm-awq), [autoawq](https://github.com/casper-hansen/AutoAWQ) or [optimum-intel](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc). Transformers supports loading models quantized with the llm-awq and autoawq libraries. This guide will show you how to load models quantized with autoawq, but the process is similar for llm-awq quantized models.

Run the command below to install autoawq


```
pip install autoawq
```

AutoAWQ downgrades Transformers to version 4.47.1. If you want to do inference with AutoAWQ, you may need to reinstall your Transformers’ version after installing AutoAWQ.

Identify an AWQ-quantized model by checking the `quant_method` key in the models [config.json](https://huggingface.co/TheBloke/zephyr-7B-alpha-AWQ/blob/main/config.json) file.


```
{
  "_name_or_path": "/workspace/process/huggingfaceh4_zephyr-7b-alpha/source",
  "architectures": [
    "MistralForCausalLM"
  ],
  ...
  ...
  ...
  "quantization_config": {
    "quant_method": "awq",
    "zero_point": true,
    "group_size": 128,
    "bits": 4,
    "version": "gemm"
  }
}
```

Load the AWQ-quantized model with [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). This automatically sets the other weights to fp16 by default for performance reasons. Use the `dtype` parameter to load these other weights in a different format.

If the model is loaded on the CPU, use the `device_map` parameter to move it to an accelerator.


```
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device
import torch

device = f"{infer_device()}:0"

model = AutoModelForCausalLM.from_pretrained(
  "TheBloke/zephyr-7B-alpha-AWQ",
  dtype=torch.float32,
  device_map=device
)
```

Use `attn_implementation` to enable [FlashAttention2](../perf_infer_gpu_one#flashattention-2) to further accelerate inference.


```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
  "TheBloke/zephyr-7B-alpha-AWQ",
  attn_implementation="flash_attention_2",
  device_map="cuda:0"
)
```

## Fused modules

Fused modules offer improved accuracy and performance. They are supported out-of-the-box for AWQ modules for [Llama](https://huggingface.co/meta-llama) and [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1) architectures, but you can also fuse AWQ modules for unsupported architectures.

Fused modules cannot be combined with other optimization techniques such as FlashAttention2.

supported architectures

unsupported architectures

Create an [AwqConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.AwqConfig) and set the parameters `fuse_max_seq_len` and `do_fuse=True` to enable fused modules. The `fuse_max_seq_len` parameter is the total sequence length and it should include the context length and the expected generation length. Set it to a larger value to be safe.

The example below fuses the AWQ modules of the [TheBloke/Mistral-7B-OpenOrca-AWQ](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ) model.


```
import torch
from transformers import AwqConfig, AutoModelForCausalLM

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=True,
)
model = AutoModelForCausalLM.from_pretrained(
  "TheBloke/Mistral-7B-OpenOrca-AWQ",
  quantization_config=quantization_config
).to(0)
```

The [TheBloke/Mistral-7B-OpenOrca-AWQ](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-AWQ) model was benchmarked with `batch_size=1` with and without fused modules.

Unfused module

| Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM) |
| --- | --- | --- | --- | --- | --- |
| 1 | 32 | 32 | 60.0984 | 38.4537 | 4.50 GB (5.68%) |
| 1 | 64 | 64 | 1333.67 | 31.6604 | 4.50 GB (5.68%) |
| 1 | 128 | 128 | 2434.06 | 31.6272 | 4.50 GB (5.68%) |
| 1 | 256 | 256 | 3072.26 | 38.1731 | 4.50 GB (5.68%) |
| 1 | 512 | 512 | 3184.74 | 31.6819 | 4.59 GB (5.80%) |
| 1 | 1024 | 1024 | 3148.18 | 36.8031 | 4.81 GB (6.07%) |
| 1 | 2048 | 2048 | 2927.33 | 35.2676 | 5.73 GB (7.23%) |

Fused module

| Batch Size | Prefill Length | Decode Length | Prefill tokens/s | Decode tokens/s | Memory (VRAM) |
| --- | --- | --- | --- | --- | --- |
| 1 | 32 | 32 | 81.4899 | 80.2569 | 4.00 GB (5.05%) |
| 1 | 64 | 64 | 1756.1 | 106.26 | 4.00 GB (5.05%) |
| 1 | 128 | 128 | 2479.32 | 105.631 | 4.00 GB (5.06%) |
| 1 | 256 | 256 | 1813.6 | 85.7485 | 4.01 GB (5.06%) |
| 1 | 512 | 512 | 2848.9 | 97.701 | 4.11 GB (5.19%) |
| 1 | 1024 | 1024 | 3044.35 | 87.7323 | 4.41 GB (5.57%) |
| 1 | 2048 | 2048 | 2715.11 | 89.4709 | 5.57 GB (7.04%) |

The speed and throughput of fused and unfused modules were also tested with the [optimum-benchmark](https://github.com/huggingface/optimum-benchmark) library.

![generate throughput per batch size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_forward_memory_plot.png) 

forward peak memory/batch size

![forward latency per batch size](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/quantization/fused_generate_throughput_plot.png) 

generate throughput/batch size

## ExLlamaV2

[ExLlamaV2](https://github.com/turboderp/exllamav2) kernels support faster prefill and decoding. Run the command below to install the latest version of autoawq with ExLlamaV2 support.


```
pip install git+https://github.com/casper-hansen/AutoAWQ.git
```

Set `version="exllama"` in [AwqConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.AwqConfig) to enable ExLlamaV2 kernels.

ExLlamaV2 is supported on AMD GPUs.


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

quantization_config = AwqConfig(version="exllama")

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
    quantization_config=quantization_config,
    device_map="auto",
)
```

## CPU

[Intel Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/cpu/latest/) is designed to enable performance optimizations on Intel hardware. Run the command below to install the latest version of autoawq with IPEX support.


```
pip install intel-extension-for-pytorch # for IPEX-GPU refer to https://intel.github.io/intel-extension-for-pytorch/xpu/2.5.10+xpu/ 
pip install git+https://github.com/casper-hansen/AutoAWQ.git
```

Set `version="ipex"` in [AwqConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.AwqConfig) to enable ExLlamaV2 kernels.


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

device = "cpu" # set to "xpu" for Intel GPU
quantization_config = AwqConfig(version="ipex")

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
    quantization_config=quantization_config,
    device_map=device,
)
```

## Resources

Run the AWQ demo [notebook](https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY#scrollTo=Wwsg6nCwoThm) for more examples of how to quantize a model, push a quantized model to the Hub, and more.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/awq.md)
