# Optimizing inference

Inference with large language models (LLMs) can be challenging because they have to store and handle billions of parameters. To load a 70B parameter [Llama 2](https://hf.co/meta-llama/Llama-2-70b-hf) model, it requires 256GB of memory for full precision weights and 128GB of memory for half-precision weights. The most powerful GPUs today - the A100 and H100 - only have 80GB of memory.

On top of the memory requirements, inference is slow because LLMs are called repeatedly to generate the next token. The input sequence increases as generation progresses, which takes longer and longer to process.

This guide will show you how to optimize LLM inference to accelerate generation and reduce memory usage.

Try out [Text Generation Inference (TGI)](https://hf.co/docs/text-generation-inference), a Hugging Face library dedicated to deploying and serving highly optimized LLMs for inference.

## Static kv-cache and torch.compile

LLMs compute key-value (kv) values for each input token, and it performs the same kv computation each time because the generated output becomes part of the input. However, performing the same kv computation every time is not very efficient.

A *kv-cache* stores the past keys and values instead of recomputing them each time. As a result, the kv-cache is dynamic and it grows with each generation step which prevents you from taking advantage of [torch.compile](./perf_torch_compile), a powerful optimization method that fuses PyTorch code into optimized kernels.

The *static kv-cache* solves this issue by pre-allocating the kv-cache size to a maximum value, so you can combine it with [torch.compile](./perf_torch_compile) for up to a 4x speed up. Your speed up may vary depending on the model size (larger models have a smaller speed up) and hardware.

Follow this [issue](https://github.com/huggingface/transformers/issues/28981) to track which models (Llama, Gemma, Mistral, etc.) support a static kv-cache and torch.compile.

Depending on your task, there are several ways you can use the static kv-cache.

1. For basic use cases, set [cache\_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) to `"static"` (recommended).
2. For multi-turn generation or a custom generation loop, initialize and handle [StaticCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.StaticCache) directly.
3. For more unique hardware or use cases, it may be better to compile the entire [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) function into a single graph.

Regardless of how you use the static kv-cache and torch.compile, left-pad your inputs with [pad\_to\_multiple\_of](https://hf.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__.pad_to_multiple_of) to a limited set of values to avoid shape-related recompilations.

1. cache\_implementation

2. StaticCache

3. compile entire generate function

1. Set the [cache\_implementation](https://hf.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.cache_implementation) to `"static"` in a models [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig).
2. Call [torch.compile](./perf_torch_compile) to compile the forward pass with the static kv-cache.


```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", dtype="auto", device_map="auto")

model.generation_config.cache_implementation = "static"

model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
input_text = "The theory of special relativity states "
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device.type)

outputs = model.generate(**input_ids)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The theory of special relativity states 1. The speed of light is constant in all inertial reference']
```

Under the hood, [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) attempts to reuse the same cache object to avoid recompilation at each call, which is critical to get the most out of [torch.compile](./perf_torch_compile). Be aware of the following to avoid triggering recompilation or if generation is slower than expected.

1. If the batch size changes or the maximum output length increases between calls, the cache is reinitialized and recompiled.
2. The first several calls of the compiled function are slower because it is being compiled.

## Decoding strategies

Decoding can also be optimized to accelerate generation. You can use a lightweight assistant model to generate candidate tokens faster than the LLM itself or you can use a variant of this decoding strategy that works especially well for input-grounded tasks.

### Speculative decoding

For a more in-depth explanation, take a look at the [Assisted Generation: a new direction toward low-latency text generation](https://hf.co/blog/assisted-generation) blog post!

For each input token, the model weights are loaded each time during the forward pass, which is slow and cumbersome when a model has billions of parameters. Speculative decoding alleviates this slowdown by using a second smaller and faster assistant model to generate candidate tokens that are verified by the larger model in a single forward pass. If the verified tokens are correct, the LLM essentially gets them for “free” without having to generate them itself. There is no degradation in accuracy because the verification forward pass ensures the same outputs are generated as if the LLM had generated them on its own.

To get the largest speed up, the assistant model should be a lot smaller than the LLM so that it can generate tokens quickly. The assistant and LLM model must also share the same tokenizer to avoid re-encoding and decoding tokens.

Speculative decoding is only supported for the greedy search and sampling decoding strategies, and it doesn’t support batched inputs.

Enable speculative decoding by loading an assistant model and passing it to [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).

greedy search

sampling


```
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device
import torch

device = infer_device()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("Einstein's theory of relativity states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", dtype="auto").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Einstein's theory of relativity states that the speed of light is constant.    "]
```

### Prompt lookup decoding

Prompt lookup decoding is a variant of speculative decoding that is also compatible with greedy search and sampling. Prompt lookup works especially well for input-grounded tasks - such as summarization - where there is often overlapping words between the prompt and output. These overlapping n-grams are used as the LLM candidate tokens.

To enable prompt lookup decoding, specify the number of tokens that should be overlapping in the [prompt\_lookup\_num\_tokens](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.prompt_lookup_num_tokens) parameter. Then pass this parameter to [generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate).

greedy decoding

sampling


```
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device
import torch

device = infer_device()

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
inputs = tokenizer("The second law of thermodynamics states", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", dtype="auto").to(device)
assistant_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
outputs = model.generate(**inputs, prompt_lookup_num_tokens=3)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['The second law of thermodynamics states that entropy increases with temperature.      ']
```

## Attention

A known issue with transformer models is that the self-attention mechanism grows quadratically in compute and memory with the number of input tokens. This limitation is only magnified in LLMs which handles much longer sequences. To address this, try FlashAttention2 or PyTorch’s scaled dot product attention (SDPA), which are more memory efficient attention implementations.

### FlashAttention-2

FlashAttention and [FlashAttention-2](./perf_infer_gpu_one#flashattention-2) break up the attention computation into smaller chunks and reduces the number of intermediate read/write operations to the GPU memory to speed up inference. FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead.

To use FlashAttention-2, set [attn\_implementation](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.PreTrainedModel.from_pretrained.attn_implementation) to `"flash_attention_2"` in [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) or set with `model.set_attention_implementation("flash_attention_2")` to dynamically update the [attention interface](./attention_interface) after the model is loaded.


```
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Change the model's attention dynamically after loading
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=quant_config,
    dtype=torch.bfloat16
)
model.set_attention_implementation("flash_attention_2")
```

### PyTorch scaled dot product attention

Scaled dot product attention (SDPA) is automatically enabled in PyTorch 2.0 and it supports FlashAttention, xFormers, and PyTorch’s C++ implementation. SDPA chooses the most performant attention algorithm if you’re using a CUDA backend. For other backends, SDPA defaults to the PyTorch C++ implementation.

SDPA automatically supports FlashAttention-2 as long as you have the latest PyTorch version installed.

Use the [torch.nn.attention.sdpa\_kernel](https://pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) context manager to explicitly enable or disable any of the four attention algorithms. For example, use `SDPBackend.FLASH_ATTENTION` to enable FlashAttention.


```
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    dtype=torch.bfloat16,
)

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    outputs = model.generate(**inputs)
```

## Quantization

Quantization reduces the size of model weights by storing them in a lower precision. This translates to lower memory usage and makes loading LLMs for inference more accessible if you’re constrained by GPU memory.

If you aren’t limited by your GPU, you don’t necessarily need to quantize your model because it can increase latency slightly (except for AWQ and fused AWQ modules) due to the extra step required to quantize and dequantize the weights.

There are many quantization libraries (see the [Quantization](./quantization) guide for more details) available, such as Quanto, AQLM, VPTQ, AWQ, and AutoGPTQ. Feel free to try them out and see which one works best for your use case. We also recommend reading the [Overview of natively supported quantization schemes in 🤗 Transformers](https://hf.co/blog/overview-quantization-transformers) blog post which compares AutoGPTQ and bitsandbytes.

Use the Model Memory Calculator below to estimate and compare how much memory is required to load a model. For example, try estimating the memory required to load [Mistral-7B-v0.1](https://hf.co/mistralai/Mistral-7B-v0.1).

To load a model in half-precision, set the [dtype](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.PreTrainedModel.from_pretrained.dtype) parameter in [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel.from_pretrained) to `torch.bfloat16`. This requires 13.74GB of memory.


```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", dtype=torch.bfloat16, device_map="auto",
)
```

To load a quantized model (8-bit or 4-bit), try [bitsandbytes](https://hf.co/docs/bitsandbytes) and set the [load\_in\_4bit](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.BitsAndBytesConfig.load_in_4bit) or [load\_in\_8bit](https://hf.co/docs/transformers/main/en/main_classes/text_generation#transformers.BitsAndBytesConfig.load_in_8bit) parameters to `True`. Loading the model in 8-bits only requires 6.87 GB of memory.


```
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=quant_config, device_map="auto"
)
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/llm_optims.md)
