*This model was released on 2023-04-09 and added to Hugging Face Transformers on 2023-07-17.*

# Bark

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat)

## Overview

[Bark](https://huggingface.co/suno/bark) is a transformer-based text-to-speech model proposed by Suno AI in [suno-ai/bark](https://github.com/suno-ai/bark).

Bark is made of 4 main models:

* [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel) (also referred to as the ‘text’ model): a causal auto-regressive transformer model that takes as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
* [BarkCoarseModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseModel) (also referred to as the ‘coarse acoustics’ model): a causal autoregressive transformer, that takes as input the results of the [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel) model. It aims at predicting the first two audio codebooks necessary for EnCodec.
* [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel) (the ‘fine acoustics’ model), this time a non-causal autoencoder transformer, which iteratively predicts the last codebooks based on the sum of the previous codebooks embeddings.
* having predicted all the codebook channels from the [EncodecModel](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel), Bark uses it to decode the output audio array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the output sound according to specific predefined voice.

This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe) and [Sanchit Gandhi (sanchit-gandhi)](https://github.com/sanchit-gandhi).
The original code can be found [here](https://github.com/suno-ai/bark).

### Optimizing Bark

Bark can be optimized with just a few extra lines of code, which **significantly reduces its memory footprint** and **accelerates inference**.

#### Using half-precision

You can speed up inference and reduce memory footprint by 50% simply by loading the model in half-precision.


```
from transformers import BarkModel, infer_device
import torch

device = infer_device()
model = BarkModel.from_pretrained("suno/bark-small", dtype=torch.float16).to(device)
```

#### Using CPU offload

As mentioned above, Bark is made up of 4 sub-models, which are called up sequentially during audio generation. In other words, while one sub-model is in use, the other sub-models are idle.

If you’re using a CUDA GPU or Intel XPU, a simple solution to benefit from an 80% reduction in memory footprint is to offload the submodels from device to CPU when they’re idle. This operation is called *CPU offloading*. You can use it with one line of code as follows:


```
model.enable_cpu_offload()
```

Note that 🤗 Accelerate must be installed before using this feature. [Here’s how to install it.](https://huggingface.co/docs/accelerate/basic_tutorials/install)

#### Using Better Transformer

Better Transformer is an 🤗 Optimum feature that performs kernel fusion under the hood. You can gain 20% to 30% in speed with zero performance degradation. It only requires one line of code to export the model to 🤗 Better Transformer:


```
model =  model.to_bettertransformer()
```

Note that 🤗 Optimum must be installed before using this feature. [Here’s how to install it.](https://huggingface.co/docs/optimum/installation)

#### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization.

##### Installation

First, check whether your hardware is compatible with Flash Attention 2. The latest list of compatible hardware can be found in the [official documentation](https://github.com/Dao-AILab/flash-attention#installation-and-features). If your hardware is not compatible with Flash Attention 2, you can still benefit from attention kernel optimisations through Better Transformer support covered [above](https://huggingface.co/docs/transformers/main/en/model_doc/bark#using-better-transformer).

Next, [install](https://github.com/Dao-AILab/flash-attention#installation-and-features) the latest version of Flash Attention 2:


```
pip install -U flash-attn --no-build-isolation
```

##### Usage

To load a model using Flash Attention 2, we can pass the `attn_implementation="flash_attention_2"` flag to [`.from_pretrained`](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained). We’ll also load the model in half-precision (e.g. `torch.float16`), since it results in almost no degradation to audio quality but significantly lower memory usage and faster inference:


```
model = BarkModel.from_pretrained("suno/bark-small", dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
```

##### Performance comparison

The following diagram shows the latency for the native attention implementation (no optimisation) against Better Transformer and Flash Attention 2. In all cases, we generate 400 semantic tokens on a 40GB A100 GPU with PyTorch 2.1. Flash Attention 2 is also consistently faster than Better Transformer, and its performance improves even more as batch sizes increase:

![](https://huggingface.co/datasets/ylacombe/benchmark-comparison/resolve/main/Bark%20Optimization%20Benchmark.png)

To put this into perspective, on an NVIDIA A100 and when generating 400 semantic tokens with a batch size of 16, you can get 17 times the [throughput](https://huggingface.co/blog/optimizing-bark#throughput) and still be 2 seconds faster than generating sentences one by one with the native model implementation. In other words, all the samples will be generated 17 times faster.

At batch size 8, on an NVIDIA A100, Flash Attention 2 is also 10% faster than Better Transformer, and at batch size 16, 25%.

#### Combining optimization techniques

You can combine optimization techniques, and use CPU offload, half-precision and Flash Attention 2 (or 🤗 Better Transformer) all at once.


```
from transformers import BarkModel, infer_device
import torch

device = infer_device()

# load in fp16 and use Flash Attention 2
model = BarkModel.from_pretrained("suno/bark-small", dtype=torch.float16, attn_implementation="flash_attention_2").to(device)

# enable CPU offload
model.enable_cpu_offload()
```

Find out more on inference optimization techniques [here](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

### Usage tips

Suno offers a library of voice presets in a number of languages [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c).
These presets are also uploaded in the hub [here](https://huggingface.co/suno/bark-small/tree/main/speaker_embeddings) or [here](https://huggingface.co/suno/bark/tree/main/speaker_embeddings).


```
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark")
>>> model = BarkModel.from_pretrained("suno/bark")

>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

Bark can generate highly realistic, **multilingual** speech as well as other audio - including music, background noise and simple sound effects.


```
>>> # Multilingual speech - simplified Chinese
>>> inputs = processor("惊人的！我会说中文")

>>> # Multilingual speech - French - let's use a voice_preset as well
>>> inputs = processor("Incroyable! Je peux générer du son.", voice_preset="fr_speaker_5")

>>> # Bark can also generate music. You can help it out by adding music notes around your lyrics.
>>> inputs = processor("♪ Hello, my dog is cute ♪")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

The model can also produce **nonverbal communications** like laughing, sighing and crying.


```
>>> # Adding non-speech cues to the input text
>>> inputs = processor("Hello uh ... [clears throat], my dog is cute [laughter]")

>>> audio_array = model.generate(**inputs)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

To save the audio, simply take the sample rate from the model config and some scipy utility:


```
>>> from scipy.io.wavfile import write as write_wav

>>> # save audio to disk, but first take the sample rate from the model config
>>> sample_rate = model.generation_config.sample_rate
>>> write_wav("bark_generation.wav", sample_rate, audio_array)
```

## BarkConfig

### class transformers.BarkConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/configuration_bark.py#L183)

( semantic\_config: typing.Optional[dict] = None coarse\_acoustics\_config: typing.Optional[dict] = None fine\_acoustics\_config: typing.Optional[dict] = None codec\_config: typing.Optional[dict] = None initializer\_range = 0.02 \*\*kwargs  )

Parameters

* **semantic\_config** ([BarkSemanticConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticConfig), *optional*) —
  Configuration of the underlying semantic sub-model.
* **coarse\_acoustics\_config** ([BarkCoarseConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseConfig), *optional*) —
  Configuration of the underlying coarse acoustics sub-model.
* **fine\_acoustics\_config** ([BarkFineConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineConfig), *optional*) —
  Configuration of the underlying fine acoustics sub-model.
* **codec\_config** ([AutoConfig](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoConfig), *optional*) —
  Configuration of the underlying codec sub-model.
* **Example** —
* **```python** —
  > > > from transformers import (
* **…** BarkSemanticConfig, —
* **…** BarkCoarseConfig, —
* **…** BarkFineConfig, —
* **…** BarkModel, —
* **…** BarkConfig, —
* **…** AutoConfig, —
* **…** ) —

This is the configuration class to store the configuration of a [BarkModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkModel). It is used to instantiate a Bark
model according to the specified sub-models configurations, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark
[suno/bark](https://huggingface.co/suno/bark) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_sub\_model\_configs

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/configuration_bark.py#L279)

( semantic\_config: BarkSemanticConfig coarse\_acoustics\_config: BarkCoarseConfig fine\_acoustics\_config: BarkFineConfig codec\_config: PretrainedConfig \*\*kwargs  ) → [BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)

Returns

[BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)

An instance of a configuration object

Instantiate a [BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig) (or a derived class) from bark sub-models configuration.

## BarkProcessor

### class transformers.BarkProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/processing_bark.py#L36)

( tokenizer speaker\_embeddings = None  )

Parameters

* **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) —
  An instance of [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
* **speaker\_embeddings** (`dict[dict[str]]`, *optional*) —
  Optional nested speaker embeddings dictionary. The first level contains voice preset names (e.g
  `"en_speaker_4"`). The second level contains `"semantic_prompt"`, `"coarse_prompt"` and `"fine_prompt"`
  embeddings. The values correspond to the path of the corresponding `np.ndarray`. See
  [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) for
  a list of `voice_preset_names`.

Constructs a Bark processor which wraps a text tokenizer and optional Bark voice presets into a single processor.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/processing_bark.py#L226)

( text = None voice\_preset = None return\_tensors = 'pt' max\_length = 256 add\_special\_tokens = False return\_attention\_mask = True return\_token\_type\_ids = False \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **voice\_preset** (`str`, `dict[np.ndarray]`) —
  The voice preset, i.e the speaker embeddings. It can either be a valid voice\_preset name, e.g
  `"en_speaker_1"`, or directly a dictionary of `np.ndarray` embeddings for each submodel of `Bark`. Or
  it can be a valid file name of a local `.npz` single voice preset.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors of a particular framework. Acceptable values are:
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return NumPy `np.ndarray` objects.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) object containing the output of the `tokenizer`.
If a voice preset is provided, the returned object will include a `"history_prompt"` key
containing a [BatchFeature](/docs/transformers/v4.56.2/en/main_classes/image_processor#transformers.BatchFeature), i.e the voice preset with the right tensors type.

Main method to prepare for the model one or several sequences(s). This method forwards the `text` and `kwargs`
arguments to the AutoTokenizer’s `__call__()` to encode the text. The method also proposes a
voice preset which is a dictionary of arrays that conditions `Bark`’s output. `kwargs` arguments are forwarded
to the tokenizer and to `cached_file` method if `voice_preset` is a valid filename.

#### from\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/processing_bark.py#L66)

( pretrained\_processor\_name\_or\_path speaker\_embeddings\_dict\_path = 'speaker\_embeddings\_path.json' \*\*kwargs  )

Parameters

* **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) —
  This can be either:
  + a string, the *model id* of a pretrained [BarkProcessor](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkProcessor) hosted inside a model repo on
    huggingface.co.
  + a path to a *directory* containing a processor saved using the [save\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkProcessor.save_pretrained)
    method, e.g., `./my_model_directory/`.
* **speaker\_embeddings\_dict\_path** (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`) —
  The name of the `.json` file containing the speaker\_embeddings dictionary located in
  `pretrained_model_name_or_path`. If `None`, no speaker\_embeddings is loaded.
* \***\*kwargs** —
  Additional keyword arguments passed along to both
  `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.

Instantiate a Bark processor associated with a pretrained model.

#### save\_pretrained

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/processing_bark.py#L122)

( save\_directory speaker\_embeddings\_dict\_path = 'speaker\_embeddings\_path.json' speaker\_embeddings\_directory = 'speaker\_embeddings' push\_to\_hub: bool = False \*\*kwargs  )

Parameters

* **save\_directory** (`str` or `os.PathLike`) —
  Directory where the tokenizer files and the speaker embeddings will be saved (directory will be created
  if it does not exist).
* **speaker\_embeddings\_dict\_path** (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`) —
  The name of the `.json` file that will contains the speaker\_embeddings nested path dictionary, if it
  exists, and that will be located in `pretrained_model_name_or_path/speaker_embeddings_directory`.
* **speaker\_embeddings\_directory** (`str`, *optional*, defaults to `"speaker_embeddings/"`) —
  The name of the folder in which the speaker\_embeddings arrays will be saved.
* **push\_to\_hub** (`bool`, *optional*, defaults to `False`) —
  Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
  repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
  namespace).
* **kwargs** —
  Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) method.

Saves the attributes of this processor (tokenizer…) in the specified directory so that it can be reloaded
using the [from\_pretrained()](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkProcessor.from_pretrained) method.

## BarkModel

### class transformers.BarkModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L1349)

( config  )

Parameters

* **config** ([BarkModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The full Bark model, a text-to-speech model composed of 4 sub-models:

* [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel) (also referred to as the ‘text’ model): a causal auto-regressive transformer model that
  takes
  as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
* [BarkCoarseModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseModel) (also referred to as the ‘coarse acoustics’ model), also a causal autoregressive transformer,
  that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary
  to `encodec`.
* [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel) (the ‘fine acoustics’ model), this time a non-causal autoencoder transformer, which iteratively
  predicts the last codebooks based on the sum of the previous codebooks embeddings.
* having predicted all the codebook channels from the [EncodecModel](/docs/transformers/v4.56.2/en/model_doc/encodec#transformers.EncodecModel), Bark uses it to decode the output audio
  array.

It should be noted that each of the first three modules can support conditional speaker embeddings to condition the
output sound according to specific predefined voice.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L1465)

( input\_ids: typing.Optional[torch.Tensor] = None history\_prompt: typing.Optional[dict[str, torch.Tensor]] = None return\_output\_lengths: typing.Optional[bool] = None \*\*kwargs  ) → By default

Parameters

* **input\_ids** (`Optional[torch.Tensor]` of shape (batch\_size, seq\_len), *optional*) —
  Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the
  longest generation among the batch.
* **history\_prompt** (`Optional[dict[str,torch.Tensor]]`, *optional*) —
  Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.
* **kwargs** (*optional*) — Remaining dictionary of keyword arguments. Keyword arguments are of two types:
  + Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
  + With a *semantic\_*, *coarse\_*, *fine\_* prefix, they will be input for the `generate` method of the
    semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.

  This means you can, for example, specify a generation strategy for all sub-models except one.
* **return\_output\_lengths** (`bool`, *optional*) —
  Whether or not to return the waveform lengths. Useful when batching.

Returns

By default

* **audio\_waveform** (`torch.Tensor` of shape (batch\_size, seq\_len)): Generated audio waveform.
  When `return_output_lengths=True`:
  Returns a tuple made of:
* **audio\_waveform** (`torch.Tensor` of shape (batch\_size, seq\_len)): Generated audio waveform.
* **output\_lengths** (`torch.Tensor` of shape (batch\_size)): The length of each waveform in the batch

Generates audio from an input prompt and an additional optional `Bark` speaker prompt.

Example:


```
>>> from transformers import AutoProcessor, BarkModel

>>> processor = AutoProcessor.from_pretrained("suno/bark-small")
>>> model = BarkModel.from_pretrained("suno/bark-small")

>>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
>>> voice_preset = "v2/en_speaker_6"

>>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)

>>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
>>> audio_array = audio_array.cpu().numpy().squeeze()
```

#### enable\_cpu\_offload

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L1389)

( accelerator\_id: typing.Optional[int] = 0 \*\*kwargs  )

Parameters

* **accelerator\_id** (`int`, *optional*, defaults to 0) —
  accelerator id on which the sub-models will be loaded and offloaded. This argument is deprecated.
* **kwargs** (`dict`, *optional*) —
  additional keyword arguments:
  `gpu_id`: accelerator id on which the sub-models will be loaded and offloaded.

Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
method moves one whole sub-model at a time to the accelerator when it is used, and the sub-model remains in accelerator until the next sub-model runs.

## BarkSemanticModel

### class transformers.BarkSemanticModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L592)

( config  )

Parameters

* **config** ([BarkCausalModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCausalModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark semantic (or text) model. It shares the same architecture as the coarse model.
It is a GPT-2 like autoregressive model with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L437)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None input\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **input\_embeds** (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
  Here, due to `Bark` particularities, if `past_key_values` is used, `input_embeds` will be ignored and you
  have to use `input_ids`. If `past_key_values` is not used and `use_cache` is set to `True`, `input_embeds`
  is used in priority instead of `input_ids`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BarkCausalModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## BarkCoarseModel

### class transformers.BarkCoarseModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L703)

( config  )

Parameters

* **config** ([`[BarkCausalModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCausalModel)`]) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark coarse acoustics model.
It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a
language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L437)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None input\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **input\_embeds** (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
  Here, due to `Bark` particularities, if `past_key_values` is used, `input_embeds` will be ignored and you
  have to use `input_ids`. If `past_key_values` is not used and `use_cache` is set to `True`, `input_embeds`
  is used in priority instead of `input_ids`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BarkCausalModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## BarkFineModel

### class transformers.BarkFineModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L924)

( config  )

Parameters

* **config** ([BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and
language modeling heads, one for each codebook.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L1071)

( codebook\_idx: int input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None input\_embeds: typing.Optional[torch.Tensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **codebook\_idx** (`int`) —
  Index of the codebook that will be predicted.
* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  NOT IMPLEMENTED YET.
* **input\_embeds** (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. If
  `past_key_values` is used, optionally only the last `input_embeds` have to be input (see
  `past_key_values`). This is useful if you want more control over how to convert `input_ids` indices into
  associated vectors than the model’s internal embedding lookup matrix.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Masked language modeling (MLM) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## BarkCausalModel

### class transformers.BarkCausalModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L376)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/modeling_bark.py#L437)

( input\_ids: typing.Optional[torch.Tensor] = None past\_key\_values: typing.Optional[tuple[torch.FloatTensor]] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.Tensor] = None head\_mask: typing.Optional[torch.Tensor] = None labels: typing.Optional[torch.LongTensor] = None input\_embeds: typing.Optional[torch.Tensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **past\_key\_values** (`tuple[torch.FloatTensor]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **input\_embeds** (`torch.FloatTensor` of shape `(batch_size, input_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
  Here, due to `Bark` particularities, if `past_key_values` is used, `input_embeds` will be ignored and you
  have to use `input_ids`. If `past_key_values` is not used and `use_cache` is set to `True`, `input_embeds`
  is used in priority instead of `input_ids`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **cache\_position** (`torch.Tensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.

Returns

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BarkConfig](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [BarkCausalModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCausalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## BarkCoarseConfig

### class transformers.BarkCoarseConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/configuration_bark.py#L144)

( block\_size = 1024 input\_vocab\_size = 10048 output\_vocab\_size = 10048 num\_layers = 12 num\_heads = 12 hidden\_size = 768 dropout = 0.0 bias = True initializer\_range = 0.02 use\_cache = True \*\*kwargs  )

Parameters

* **block\_size** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **input\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [BarkCoarseModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseModel). Defaults to 10\_048 but should be carefully thought with
  regards to the chosen sub-model.
* **output\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
  by the: `output_ids` when passing forward a [BarkCoarseModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseModel). Defaults to 10\_048 but should be carefully thought
  with regards to the chosen sub-model.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the given sub-model.
* **num\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer architecture.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the architecture.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use bias in the linear layers and layer norm layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [BarkCoarseModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkCoarseModel). It is used to instantiate the model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import BarkCoarseConfig, BarkCoarseModel

>>> # Initializing a Bark sub-module style configuration
>>> configuration = BarkCoarseConfig()

>>> # Initializing a model (with random weights) from the suno/bark style configuration
>>> model = BarkCoarseModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## BarkFineConfig

### class transformers.BarkFineConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/configuration_bark.py#L172)

( tie\_word\_embeddings = True n\_codes\_total = 8 n\_codes\_given = 1 \*\*kwargs  )

Parameters

* **block\_size** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **input\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel). Defaults to 10\_048 but should be carefully thought with
  regards to the chosen sub-model.
* **output\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
  by the: `output_ids` when passing forward a [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel). Defaults to 10\_048 but should be carefully thought
  with regards to the chosen sub-model.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the given sub-model.
* **num\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer architecture.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the architecture.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use bias in the linear layers and layer norm layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **n\_codes\_total** (`int`, *optional*, defaults to 8) —
  The total number of audio codebooks predicted. Used in the fine acoustics sub-model.
* **n\_codes\_given** (`int`, *optional*, defaults to 1) —
  The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics
  sub-models.

This is the configuration class to store the configuration of a [BarkFineModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkFineModel). It is used to instantiate the model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import BarkFineConfig, BarkFineModel

>>> # Initializing a Bark sub-module style configuration
>>> configuration = BarkFineConfig()

>>> # Initializing a model (with random weights) from the suno/bark style configuration
>>> model = BarkFineModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## BarkSemanticConfig

### class transformers.BarkSemanticConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/bark/configuration_bark.py#L121)

( block\_size = 1024 input\_vocab\_size = 10048 output\_vocab\_size = 10048 num\_layers = 12 num\_heads = 12 hidden\_size = 768 dropout = 0.0 bias = True initializer\_range = 0.02 use\_cache = True \*\*kwargs  )

Parameters

* **block\_size** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **input\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel). Defaults to 10\_048 but should be carefully thought with
  regards to the chosen sub-model.
* **output\_vocab\_size** (`int`, *optional*, defaults to 10\_048) —
  Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
  by the: `output_ids` when passing forward a [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel). Defaults to 10\_048 but should be carefully thought
  with regards to the chosen sub-model.
* **num\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the given sub-model.
* **num\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer architecture.
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the “intermediate” (often named feed-forward) layer in the architecture.
* **dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **bias** (`bool`, *optional*, defaults to `True`) —
  Whether or not to use bias in the linear layers and layer norm layers.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [BarkSemanticModel](/docs/transformers/v4.56.2/en/model_doc/bark#transformers.BarkSemanticModel). It is used to instantiate the model
according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import BarkSemanticConfig, BarkSemanticModel

>>> # Initializing a Bark sub-module style configuration
>>> configuration = BarkSemanticConfig()

>>> # Initializing a model (with random weights) from the suno/bark style configuration
>>> model = BarkSemanticModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bark.md)
