# GPU

GPUs are the standard hardware for machine learning because they’re optimized for memory bandwidth and parallelism. With the increasing sizes of modern models, it’s more important than ever to make sure GPUs are capable of efficiently handling and delivering the best possible performance.

This guide will demonstrate a few ways to optimize inference on a GPU. The optimization methods shown below can be combined with each other to achieve even better performance, and they also work for distributed GPUs.

## bitsandbytes

[bitsandbytes](https://hf.co/docs/bitsandbytes/index) is a quantization library that supports 8-bit and 4-bit quantization. Quantization represents weights in a lower precision compared to the original full precision format. It reduces memory requirements and makes it easier to fit large model into memory.

Make sure bitsandbytes and Accelerate are installed first.

```
pip install bitsandbytes accelerate
```

8-bit

4-bit

For text generation with 8-bit quantization, you should use [generate()](/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) instead of the high-level [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) API. The [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) returns slower performance because it isn’t optimized for 8-bit models, and some sampling strategies (nucleus sampling) also aren’t supported.

Set up a [BitsAndBytesConfig](/docs/transformers/main/en/main_classes/quantization#transformers.BitsAndBytesConfig) and set `load_in_8bit=True` to load a model in 8-bit precision. The [BitsAndBytesConfig](/docs/transformers/main/en/main_classes/quantization#transformers.BitsAndBytesConfig) is passed to the `quantization_config` parameter in [from\_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained).

Allow Accelerate to automatically distribute the model across your available hardware by setting [device\_map=“auto”](https://hf.co/docs/accelerate/concept_guides/big_model_inference#designing-a-device-map).

Place all inputs on the same device as the model.

```
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config)

prompt = "Hello, my llama is cute"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```

For distributed setups, use the `max_memory` parameter to create a mapping of the amount of memory to allocate to each GPU. The example below distributes 16GB of memory to the first GPU and 16GB of memory to the second GPU.

```
max_memory_mapping = {0: "16GB", 1: "16GB"}
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B", device_map="auto", quantization_config=quantization_config, max_memory=max_memory_mapping
)
```

Learn in more detail the concepts underlying 8-bit quantization in the [Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://hf.co/blog/hf-bitsandbytes-integration) blog post.

## Optimum

[Optimum](https://hf.co/docs/optimum/en/index) is a Hugging Face library focused on optimizing model performance across various hardware. It supports [ONNX Runtime](https://onnxruntime.ai/docs/) (ORT), a model accelerator, for a wide range of hardware and frameworks including NVIDIA GPUs and AMD GPUs that use the [ROCm](https://www.amd.com/en/products/software/rocm.html) stack.

ORT uses optimization techniques that fuse common operations into a single node and constant folding to reduce the number of computations. ORT also places the most computationally intensive operations on the GPU and the rest on the CPU to intelligently distribute the workload between the two devices.

Optimum provides the `ORTModel` class for loading ONNX models. Set the `provider` parameter according to the table below.

| provider | hardware |
| --- | --- |
| [CUDAExecutionProvider](https://hf.co/docs/optimum/main/en/onnxruntime/usage_guides/gpu#cudaexecutionprovider) | CUDA-enabled GPUs |
| [ROCMExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/amdgpu) | AMD Instinct, Radeon Pro, Radeon GPUs |
| [TensorrtExecutionProvider](https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#tensorrtexecutionprovider) | TensorRT |

For example, load the [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://hf.co/optimum/roberta-base-squad2) checkpoint for sequence classification. This checkpoint contains a [model.onnx](https://hf.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/main/onnx/model.onnx) file. If a checkpoint doesn’t have a `model.onnx` file, set `export=True` to convert a checkpoint on the fly to the ONNX format.

```
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  #export=True,
  provider="CUDAExecutionProvider",
)
```

Now you can use the model for inference in a [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline).

```
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
pipeline = pipeline(task="text-classification", model=ort_model, tokenizer=tokenizer, device="cuda:0")
result = pipeline("Both the music and visual were astounding, not to mention the actors performance.")
```

Learn more details about using ORT with Optimum in the [Accelerated inference on NVIDIA GPUs](https://hf.co/docs/optimum/onnxruntime/usage_guides/gpu#accelerated-inference-on-nvidia-gpus) and [Accelerated inference on AMD GPUs](https://hf.co/docs/optimum/onnxruntime/usage_guides/amdgpu#accelerated-inference-on-amd-gpus) guides.

## Scaled dot product attention (SDPA)

PyTorch’s [torch.nn.functional.scaled\_dot\_product\_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA) is a native implementation of the scaled dot product attention mechanism. SDPA is a more efficient and optimized version of the attention mechanism used in transformer models.

There are three supported implementations available.

* [FlashAttention2](https://github.com/Dao-AILab/flash-attention) only supports models with the fp16 or bf16 torch type. Make sure to cast your model to the appropriate type first.
* [xFormers](https://github.com/facebookresearch/xformers) or Memory-Efficient Attention is able to support models with the fp32 torch type.
* C++ implementation of scaled dot product attention

SDPA is used by default for PyTorch v2.1.1. and greater when an implementation is available. You could explicitly enable SDPA by setting `attn_implementation="sdpa"` in [from\_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) though. Certain attention parameters, such as `output_attentions=True`, are unsupported and returns a warning that Transformers will fall back to the (slower) eager implementation.

Refer to the [AttentionInterface](./attention_interface) guide to learn how to change the attention implementation after loading a model.

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", attn_implementation="sdpa")

# Change the model's attention dynamically after loading it
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto")
model.set_attention_implementation("sdpa")
```

SDPA selects the most performant implementation available, but you can also explicitly select an implementation with [torch.nn.attention.sdpa\_kernel](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) as a context manager. The example below shows how to enable the FlashAttention2 implementation with `enable_flash=True`.

```
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto")

input_text = "Hello, my llama is cute"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you encounter the following `RuntimeError`, try installing the nightly version of PyTorch which has broader coverage for FlashAttention.

```
RuntimeError: No available kernel. Aborting execution.

pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

## FlashAttention

[FlashAttention](https://github.com/Dao-AILab/flash-attention) is also available as a standalone package. It can significantly speed up inference by:

1. additionally parallelizing the attention computation over sequence length
2. partitioning the work between GPU threads to reduce communication and shared memory reads/writes between them

Install FlashAttention first for the hardware you’re using.

NVIDIA

AMD

```
pip install flash-attn --no-build-isolation
```

Enable FlashAttention2 by setting `attn_implementation="flash_attention_2"` in [from\_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) or by setting `model.set_attention_implementation("flash_attention_2")` to dynamically update the [attention interface](./attention_interface). FlashAttention2 is only supported for models with the fp16 or bf16 torch type. Make sure to cast your model to the appropriate data type first.

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", device_map="auto", dtype=torch.bfloat16, attn_implementation="flash_attention_2")
```

### Benchmarks

FlashAttention2 speeds up inference considerably especially for inputs with long sequences. However, since FlashAttention2 doesn’t support computing attention scores with padding tokens, you must manually pad and unpad the attention scores for batched inference if a sequence contains padding tokens. The downside is batched generation is slower with padding tokens.

short sequence length

long sequence length

With a relatively small sequence length, a single forward pass creates overhead leading to a small speed up. The graph below shows the expected speed up for a single forward pass with [meta-llama/Llama-7b-hf](https://hf.co/meta-llama/Llama-7b-hf) with padding.

![](https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/llama-2-small-seqlen-padding.png)

To avoid this slowdown, use FlashAttention2 without padding tokens in the sequence during training. Pack the dataset or concatenate sequences until reaching the maximum sequence length.

tiiuae/falcon-7b

meta-llama/Llama-7b-hf

The graph below shows the expected speed up for a single forward pass with [tiiuae/falcon-7b](https://hf.co/tiiuae/falcon-7b) with a sequence length of 4096 and various batch sizes without padding tokens.

![](https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/falcon-7b-inference-large-seqlen.png)

 [Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/perf_infer_gpu_one.md)
