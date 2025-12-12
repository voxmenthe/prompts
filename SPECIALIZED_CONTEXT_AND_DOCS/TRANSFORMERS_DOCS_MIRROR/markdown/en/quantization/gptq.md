# GPTQ

The [GPTQModel](https://github.com/ModelCloud/GPTQModel) and [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) implements the GPTQ algorithm, a post-training quantization technique where each row of the weight matrix is quantized independently to find a version of the weights that minimizes the error. These weights are quantized to int4, but they’re restored to fp16 on the fly during inference. This can save memory usage by 4x because the int4 weights are dequantized in a fused kernel rather than a GPU’s global memory. Inference is also faster because a lower bitwidth takes less time to communicate.

AutoGPTQ is likely to be deprecated in the future due to lack of continued support for new models and features. See the [GPTQModel](#gptqmodel) section for more details.

Install Accelerate, Transformers and Optimum first.


```
pip install --upgrade accelerate optimum transformers
```

Then run the command below to install a GPTQ library.

GPTQmodel

AutoGPTQ


```
pip install gptqmodel --no-build-isolation
```

Create a [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig) class and set the number of bits to quantize to, a dataset to calbrate the weights for quantization, and a tokenizer to prepare the dataset.


```
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
```

You can pass your own dataset as a list of strings, but it is highly recommended to use the same dataset from the GPTQ paper.


```
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
```

Load a model to quantize and pass [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig) to [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained). Set `device_map="auto"` to automatically offload the model to a CPU to help fit the model in memory, and allow the model modules to be moved between the CPU and GPU for quantization.


```
quantized_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", device_map="auto", quantization_config=gptq_config)
```

If you’re running out of memory because a dataset is too large (disk offloading is not supported), try passing the `max_memory` parameter to allocate the amount of memory to use on your device (GPU and CPU).


```
quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
    max_memory={0: "30GiB", 1: "46GiB", "cpu": "30GiB"},
    quantization_config=gptq_config
)
```

Depending on your hardware, it can take some time to quantize a model from scratch. It can take ~5 minutes to quantize the [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) model on a free-tier Google Colab GPU, but it’ll take ~4 hours to quantize a 175B parameter model on a NVIDIA A100. Before you quantize a model, it is a good idea to check the Hub if a GPTQ-quantized version of the model already exists.

Once a model is quantized, you can use [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) to push the model and tokenizer to the Hub where it can be easily shared and accessed. This saves the [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig).


```
quantized_model.push_to_hub("opt-125m-gptq")
tokenizer.push_to_hub("opt-125m-gptq")
```

[save\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) saves a quantized model locally. If the model was quantized with the `device_map` parameter, make sure to move the entire model to a GPU or CPU before saving it. The example below saves the model on a CPU.


```
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

# if quantized with device_map set
quantized_model.to("cpu")
quantized_model.save_pretrained("opt-125m-gptq")
```

Reload a quantized model with [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained), and set `device_map="auto"` to automatically distribute the model on all available GPUs to load the model faster without using more memory than needed.


```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto")
```

## Marlin

[Marlin](https://github.com/IST-DASLab/marlin) is a 4-bit only CUDA GPTQ kernel, highly optimized for the NVIDIA A100 GPU (Ampere) architecture. Loading, dequantization, and execution of post-dequantized weights are highly parallelized, offering a substantial inference improvement versus the original CUDA GPTQ kernel. Marlin is only available for quantized inference and does not support model quantization.

Marlin inference can be activated with the `backend` parameter in [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig).


```
from transformers import AutoModelForCausalLM, GPTQConfig

model = AutoModelForCausalLM.from_pretrained("{your_username}/opt-125m-gptq", device_map="auto", quantization_config=GPTQConfig(bits=4, backend="marlin"))
```

## ExLlama

Only 4-bit models are supported, and we recommend deactivating the ExLlama kernels if you’re finetuning a quantized model with PEFT.

[ExLlama](https://github.com/turboderp/exllama) is a Python/C++/CUDA implementation of the [Llama](model_doc/llama) model that is designed for faster inference with 4-bit GPTQ weights (check out these [benchmarks](https://github.com/huggingface/optimum/tree/main/tests/benchmark#gptq-benchmark)). The ExLlama kernel is activated by default when you create a [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig) object.

To boost inference speed even further, use the [ExLlamaV2](https://github.com/turboderp/exllamav2) kernels by configuring the `exllama_config` parameter in [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig).


```
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, exllama_config={"version":2})
model = AutoModelForCausalLM.from_pretrained(
    "{your_username}/opt-125m-gptq",
    device_map="auto",
    quantization_config=gptq_config
)
```

The ExLlama kernels are only supported when the entire model is on the GPU. If you’re doing inference on a CPU with AutoGPTQ 0.4.2+, disable the ExLlama kernel in [GPTQConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.GPTQConfig). This overwrites the attributes related to the ExLlama kernels in the quantization config of the `config.json` file.


```
import torch
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, use_exllama=False)
model = AutoModelForCausalLM.from_pretrained(
    "{your_username}/opt-125m-gptq",
    device_map="cpu",
    quantization_config=gptq_config
)
```

## GPTQModel

It is recommended to use GPTQModel, originally a maintained fork of AutoGPTQ, because it has since diverged from AutoGTPQ with some significant features. GPTQModel has faster quantization, lower memory usage, and more accurate default quantization.

GPTQModel provides asymmetric quantization which can potentially lower quantization errors compared to symmetric quantization. It is not backward compatible with AutoGPTQ, and not all kernels (Marlin) support asymmetric quantization.

GPTQModel also has broader support for the latest LLM models, multimodal models (Qwen2-VL and Ovis1.6-VL), platforms (Linux, macOS, Windows 11), and hardware (AMD ROCm, Apple Silicon, Intel/AMD CPUs, and Intel Datacenter Max/Arc GPUs, etc.).

The Marlin kernels are also updated for A100 GPUs and other kernels are updated to include auto-padding for legacy models and models with non-uniform in/out-features.

## Resources

Run the GPTQ quantization with PEFT [notebook](https://colab.research.google.com/drive/1_TIrmuKOFhuRRiTWN94iLKUFu6ZX4ceb?usp=sharing) for a hands-on experience, and read [Making LLMs lighter with AutoGPTQ and transformers](https://huggingface.co/blog/gptq-integration) to learn more about the AutoGPTQ integration.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/gptq.md)
