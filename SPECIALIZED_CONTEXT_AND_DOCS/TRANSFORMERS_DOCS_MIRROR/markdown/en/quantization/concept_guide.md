# Quantization concepts

Quantization reduces the memory footprint and computational cost of large machine learning models like those found in the Transformers library. It achieves this by representing the model’s weights and or activations with lower-precision data types (like 8-bit integers or int8) instead of the standard 32-bit floating-point (float32).

Reducing a model’s precision offers several significant benefits:

* Smaller model size: Lower-precision data types require less storage space. An int8 model, for example, is roughly 4 times smaller than its float32 counterpart.
* Faster inference: Operations on lower-precision data types, especially integers, can be significantly faster on compatible hardware (CPUs and GPUs often have specialized instructions for int8 operations). This leads to lower latency.
* Reduced energy consumption: Faster computations and smaller memory transfers often translate to lower power usage.

The primary trade-off in quantization is *efficiency* vs. *accuracy*. Reducing precision saves resources but inevitably introduces small errors (quantization noise). The goal is to minimize this error using appropriate schemes (affine/symmetric), granularity (per-tensor/channel), and techniques (PTQ/QAT) so that the model’s performance on its target task degrades as little as possible.

The sections below cover quantization schemes, granularity, and techniques.

## Quantization schemes

The core idea is to map the range of values found in the original float32 weights and activations to the much smaller range represented by int8 (typically[−128,127][-128, 127][−128,127]).

This section covers how some quantization techniques work.

![quant_visual](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/quant_visual.png)

### Affine quantization

The most common method is *affine quantization*. For a given float32 tensor (like a layer’s weights), it finds the minimumvalminval\_{min}valmin​ and maximumvalmaxval\_{max}valmax​ values. This range[valmin,valmax][val\_{min}, val\_{max}][valmin​,valmax​] is mapped to the int8 range[qmin,qmax][q\_{min}, q\_{max}][qmin​,qmax​], which is typically[−128,127][-128, 127][−128,127].

There are two main ways to perform this mapping, *symmetric* and *asymmetric*. The choice between symmetric and asymmetric quantization determines how the float32 range is mapped to the int8 range.

* Symmetric: This method assumes the original float32 range is symmetric around zero ([−a,a][ -a, a ][−a,a] ). This range is mapped symmetrically to the int8 range, for example,[−127,127][-127, 127][−127,127]. A key characteristic is that the float32 value0.00.00.0 maps directly to the int8 value000. This only requires one parameter, the **scale (SSS )**, to define the mapping. It can simplify computations, but it might be less accurate if the original data distribution isn’t naturally centered around zero.
* Asymmetric (Affine): This method does not assume the data is centered around zero. It maps the exact range[valmin,valmax][val\_{min}, val\_{max}][valmin​,valmax​] from float32 to the full int8 range, like[−128,127][-128, 127][−128,127]. This requires two parameters, a **scale (SSS )** and a **zero-point (ZZZ )**.

scale (SSS ): A positive float32 number representing the ratio between the float32 and the int8 range.
S=valmax−valminqmax−qmin
S = \frac{val\_{max} - val\_{min}}{q\_{max} - q\_{min}}
S=qmax​−qmin​valmax​−valmin​​

zero-Point (ZZZ ): An int8 value that corresponds to the float32 value0.00.00.0.
Z=qmin−round(valminS)
Z = q\_{min} - round\left(\frac{val\_{min}}{S}\right)
Z=qmin​−round(Svalmin​​)

In symmetric quantization, Z would typically be fixed at 0.

With these parameters, a float32 value,xxx. can be quantized to int8 (qqq ) with the formula below.
q=round(xS+Z)
q = round\left(\frac{x}{S} + Z\right)
q=round(Sx​+Z)

The int8 value,qqq, can be dequantized back to approximate float32 with the formula below.
x≈S⋅(q−Z)
x \approx S \cdot (q - Z)
x≈S⋅(q−Z)

![dequant](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/dequant.png)

During inference, computations like matrix multiplication are performed using the int8 values (qqq ), and the result is dequantized back to float32 (often using a higher-precision accumulation type like int32 internally) before it is passed to the next layer.

### int4 and weight packing

![weight packing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/weight_packing.png)

int4 quantization further reduces the model size and memory usage (halving it compared to int8). The same affine or symmetric quantization principles apply, mapping the float32 range to the 16 possible values representable by int4 ([−8,7][-8, 7][−8,7] for signed int4).

A key aspect of int4 quantization is **weight packing**. Since most hardware can’t natively handle 4-bit data types in memory, two int4 values are typically packed together into a single int8 byte for storage and transfer. For example, the first value might occupy the lower 4 bits and the second value the upper 4 bits of the byte (`packed_byte = (val1 & 0x0F) | (val2 << 4)`).

int4 is still beneficial even without native int4 compute because the primary benefit comes from reduced memory bandwidth. Loading packed int4 weights (stored as int8) from memory (RAM or VRAM) to the compute units is twice as fast as loading int8 weights. For large models, memory access is often a significant bottleneck. The speed up from faster data transfer can outweigh the computational overhead of unpacking and dequantizing on the fly, leading to overall faster inference, especially in memory-bound scenarios.

However, int4 quantization typically results in a larger accuracy drop compared to int8. Advanced quantization techniques like [GPTQ](./gptq) or [AWQ](./awq) are often necessary for good performance with int4.

### FP8 Quantization (A8W8)

![fp8](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/fp8.png)

A newer datatype, 8-bit floating-point (FP8), offers another way to reduce precision while retaining more accuracy than int8 in certain scenarios. FP8 keeps the floating-point structure (sign, exponent, mantissa) but uses fewer bits.

There are two common FP8 variants.

* E4M3: 1 sign bit, 4 exponent bits, 3 mantissa bits. Offers higher precision (more mantissa bits) but a smaller dynamic range (fewer exponent bits).
* E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits. Offers a wider dynamic range but lower precision.

FP8 is used in the *A8W8* quantization scheme, which quantizes both activations (A) and weights (W) to 8-bit precision.

While int8 has broad support, efficient FP8 computation requires specific hardware capabilities found in newer GPUs like NVIDIA H100/H200/B100 and AMD Instinct MI300 series. Without native hardware acceleration, the benefits of FP8 might not be fully realized.

Transformers supports FP8 through specific backends like [FBGEMM](./fbgemm_fp8), [FineGrainedFP8](./finegrained_fp8), and [compressed-tensors](./compressed_tensors). These backends handle the underlying FP8 conversion and computation when the appropriate hardware and configurations are used.

## Granularity

Quantization parameters (SSS andZZZ) can be calculated in one of two ways.

* Per-Tensor: One set ofSSS andZZZ for the entire tensor. Simpler, but less accurate if data values vary greatly within the tensor.
* Per-Channel (or Per-Group/Block): SeparateSSS andZZZ for each channel or group. More accurate and better performance at the cost of slightly more complexity and memory.

![Granularities](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Granularities.png)

## Quantization techniques

There are two main types of quantization techniques.

* Post-Training Quantization (PTQ): Quantization is applied *after* the model is fully trained.
* Quantization-Aware Training (QAT): Quantization effects are simulated *during* training by inserting “fake quantization” ops that simulate the rounding errors of quantization. This lets the model adapt to quantization, and usually results in better accuracy, especially at lower bit-widths.

## Quantization in Transformers

Transformers integrates several quantization backends such as bitsandbytes, torchao, compressed-tensors, and more (refer to the quantization [overview](./overview) for more backends).

All backends are unified under the `HfQuantizer` API and associated `QuantizationConfig` classes. You can integrate your own custom quantization backends by implementing a custom `HfQuantizer` and `QuantizationConfig`, as shown in the [Contribution](./contribute) guide.

The typical workflow for quantization in Transformers is to:

1. Choose a quantization method suitable for your hardware and use case (see the [Overview](./overview) or [Selecting a quantization method](./selecting) guide to help you).
2. Load a pre-quantized model from the Hugging Face Hub or load a float32/float16/bfloat16 model and apply a specific quantization method with `QuantizationConfig`.

The example below demonstrates loading a 8B parameter model and quantizing it to 4-bits with bitsandbytes.


```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "meta-llama/Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Resources

To explore quantization and related performance optimization concepts more deeply, check out the following resources.

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth)
* [Introduction to Quantization cooked in 🤗 with 💗🧑‍🍳](https://huggingface.co/blog/merve/quantization)
* [EfficientML.ai Lecture 5 - Quantization Part I](https://www.youtube.com/watch?v=RP23-dRVDWM)
* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
* [Accelerating Generative AI with PyTorch Part 2: LLM Optimizations](https://pytorch.org/blog/accelerating-generative-ai-2/)

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/quantization/concept_guide.md)
