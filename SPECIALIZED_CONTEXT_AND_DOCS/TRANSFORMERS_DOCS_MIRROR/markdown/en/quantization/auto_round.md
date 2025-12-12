# AutoRound

[AutoRound](https://github.com/intel/auto-round) is an advanced quantization algorithm that delivers strong accuracy, even at 2-bit precision.
It leverages sign gradient descent to fine-tune both rounding values and min-max clipping thresholds in just 200 steps. Designed for broad compatibility, it seamlessly supports a wide range of LLMs and is actively expanding to cover more VLMs as well.
It also supports quantization and inference across multiple hardware platforms, including CPU, XPU, and CUDA.

AutoRound also offers a variety of useful features, including mixed-bit tuning and inference, lm-head quantization, support for exporting to formats like GPTQ/AWQ/GGUF, and flexible tuning recipes.
For a comprehensive overview and the latest updates, check out the AutoRound [README](https://github.com/intel/auto-round).

AutoRound was originally developed as part of the [Intel Neural Compressor](https://github.com/intel/neural-compressor), serving as a general-purpose model compression library for deep learning.
It has since evolved into a standalone library focused specifically on low-precision optimization for large language models (LLMs).
AutoRound remains fully integrated with the Intel Neural Compressor, and you can explore the repository for more details.

## Installation


```
pip install auto-round
```

## Supported Quantization Configurations

AutoRound supports several quantization configurations:

* **Int8 Weight Only**
* **Int4 Weight Only**
* **Int3 Weight Only**
* **Int2 Weight Only**
* **Mixed bits Weight only**

## Hardware Compatibility

CPU, XPU, and CUDA for both quantization and inference.

## Quantization and Serialization (offline)

Currently, only offline mode is supported to generate quantized models.

quantization cmd

quantization auto-round api

quantization auto-round-best

quantization auto-round-light

### Command Line Usage


```
auto-round \
    --model facebook/opt-125m \
    --bits 4 \
    --group_size 128 \
    --output_dir ./tmp_autoround
```

AutoRound also offer another two recipes, `auto-round-best` and `auto-round-light`, designed for optimal accuracy and improved speed, respectively.
For 2 bits, we recommend using `auto-round-best` or `auto-round`.

W4G128 Average Accuracy of 13 tasks (mmlu-pro, if\_eval, gsm8k, etc) and Time Cost Results (Testing was conducted on the Nvidia A100 80G using the version of PyTorch 2.6.0 with enable\_torch\_compile):

| Model | Qwen2.5-0.5B-Instruct | Falcon3-3B | Qwen2.5-7B-Instruct | Meta-Llama-3.1-8B-Instruct | Falcon3-10B | Qwen2.5-72B-Instruct |
| --- | --- | --- | --- | --- | --- | --- |
| 16bits | 0.4192 | 0.5203 | 0.6470 | 0.6212 | 0.6151 | 0.7229 |
| Best | **0.4137**(7m) | **0.5142**(23m) | 0.6426(58m) | **0.6116**(65m) | **0.6092**(81m) | 0.7242(575m) |
| Default | 0.4129(2m) | 0.5133(6m) | 0.6441(13m) | 0.6106(13m) | 0.6080(18m) | **0.7252**(118m) |
| Light | 0.4052(2m) | 0.5108(3m) | **0.6453**(5m) | 0.6104(6m) | 0.6063(6m) | 0.7243(37m) |

## Inference

AutoRound automatically selects the best available backend based on the installed libraries and prompts the user to install additional libraries when a better backend is found.

inference cpu

inference xpu

inference cuda

inference backend

format convert

### CPU

Supports 2, 4, and 8 bits. We recommend using intel-extension-for-pytorch (IPEX) for 4 bits inference.


```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "There is a girl who likes adventure,"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50, do_sample=False)[0]))
```

## Issues

If you encounter any issues with the transformers integration, please open an issue on
the [transformers](https://github.com/huggingface/transformers/issues) repository.  
If you encounter any issues with auto-round, please open an issue on
the [AutoRound](https://github.com/intel/auto-round/issues) repository.

## Acknowledgement

Special thanks to open-source low precision libraries such as AutoGPTQ, AutoAWQ, GPTQModel, Triton, Marlin, and ExLLaMAV2 for providing low-precision CUDA kernels, which are leveraged in AutoRound.

## Contribution

Contributions to [AutoRound](https://github.com/intel/auto-round/pulls) are welcome and greatly appreciated!
Whether it’s fixing bugs, improving documentation, adding new features, or suggesting improvements, your help is always valued.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/auto_round.md)
