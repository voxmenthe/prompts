# HIGGS

[HIGGS](https://huggingface.co/papers/2411.17525) is a zero-shot quantization algorithm that combines Hadamard preprocessing with MSE-Optimal quantization grids to achieve lower quantization error and state-of-the-art performance.

Runtime support for HIGGS is implemented through the [FLUTE](https://github.com/HanGuo97/flute) library. Only the 70B and 405B variants of Llama 3 and Llama 3.0, and the 8B and 27B variants of Gemma 2 are currently supported. HIGGS also doesn’t support quantized training and backward passes in general at the moment.

Run the command below to install FLUTE.

CUDA 12.1

CUDA 11.8


```
pip install flute-kernel
```

Create a [HiggsConfig](/docs/transformers/v4.56.2/en/main_classes/quantization#transformers.HiggsConfig) with the number of bits to quantize a model to.


```
from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=HiggsConfig(bits=4),
    device_map="auto",
)
```

Find models pre-quantized with HIGGS in the official ISTA-DASLab [collection](https://huggingface.co/collections/ISTA-DASLab/higgs-675308e432fd56b7f6dab94e).

## torch.compile

HIGGS is fully compatible with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HiggsConfig

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    quantization_config=HiggsConfig(bits=4),
    device_map="auto",
)

model = torch.compile(model)
```

Refer to the table below for a benchmark of forward passes/sec for Llama-3.1-8B-Instruct on a RTX4090.

| Batch Size | BF16 (with `torch.compile`) | HIGGS 4bit (without `torch.compile`) | HIGGS 4bit (with `torch.compile`) |
| --- | --- | --- | --- |
| 1 | 59 | 41 | 124 |
| 4 | 57 | 42 | 123 |
| 16 | 56 | 41 | 120 |

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/higgs.md)
