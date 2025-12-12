*This model was released on 2024-05-03 and added to Hugging Face Transformers on 2024-04-15.*

# Idefics2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white) ![FlashAttention](https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat) ![SDPA](https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The Idefics2 model was proposed in [What matters when building vision-language models?](https://huggingface.co/papers/2405.02246) by Léo Tronchon, Hugo Laurencon, Victor Sanh. The accompanying blog post can be found [here](https://huggingface.co/blog/idefics2).

Idefics2 is an open multimodal model that accepts arbitrary sequences of image and text inputs and produces text
outputs. The model can answer questions about images, describe visual content, create stories grounded on multiple
images, or simply behave as a pure language model without visual inputs. It improves upon IDEFICS-1, notably on
document understanding, OCR, or visual reasoning. Idefics2 is lightweight (8 billion parameters) and treats
images in their native aspect ratio and resolution, which allows for varying inference efficiency.

The abstract from the paper is the following:

*The growing interest in vision-language models (VLMs) has been driven by improvements in large language models and vision transformers. Despite the abundance of literature on this subject, we observe that critical decisions regarding the design of VLMs are often not justified. We argue that these unsupported decisions impede progress in the field by making it difficult to identify which choices improve model performance. To address this issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters. Idefics2 achieves state-of-the-art performance within its size category across various multimodal benchmarks, and is often on par with models four times its size. We release the model (base, instructed, and chat) along with the datasets created for its training.*

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/idefics2_architecture.png) Idefics2 architecture. Taken from the [original paper.](https://huggingface.co/papers/2405.02246)

This model was contributed by [amyeroberts](https://huggingface.co/amyeroberts).
The original code can be found [here](https://huggingface.co/HuggingFaceM4/idefics2).

## Usage tips

* Each sample can contain multiple images, and the number of images can vary between samples. The processor will pad the inputs to the maximum number of images in a batch for input to the model.
* The processor has a `do_image_splitting` option. If `True`, each input image will be split into 4 sub-images, and concatenated with the original to form 5 images. This is useful for increasing model performance. Make sure `processor.image_processor.do_image_splitting` is set to `False` if the model was not trained with this option.
* `text` passed to the processor should have the `<image>` tokens where the images should be inserted. And `<end_of_utterance>` at the end of each utterance if the text is a chat message.
* The processor has its own `apply_chat_template` method to convert chat messages to text that can then be passed as `text` to the processor.

Example of how to use the processor on chat messages:


```
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration, infer_device
import torch

device = infer_device()

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
}]

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

# at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
text = processor.apply_chat_template(messages, add_generation_prompt=True)
print(text)
# 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant:'

inputs = processor(images=images, text=text, return_tensors="pt").to(model.device)

generated_text = model.generate(**inputs, max_new_tokens=500)
generated_text = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
print("Generated text:", generated_text)
```

* During training, it’s important to determine which tokens the model should not learn. For Idefics2, this typically comes down to the image and padding tokens. This means that one can create the labels as follows:


```
import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration, infer_device
import torch

url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
url_2 = "http://images.cocodataset.org/val2017/000000219578.jpg"

image_1 = Image.open(requests.get(url_1, stream=True).raw)
image_2 = Image.open(requests.get(url_2, stream=True).raw)
images = [image_1, image_2]

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "What’s the difference between these two images?"},
        {"type": "image"},
        {"type": "image"},
    ],
},
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The difference is that one image is about dogs and the other one about cats."},
    ],
}]

device = infer_device()

processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b")
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b")
model.to(device)

text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(images=images, text=text, return_tensors="pt").to(model.device)

labels = inputs.input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
labels[labels == model.config.image_token_id] = -100

inputs["labels"] = labels

outputs = model(**inputs)
loss = outputs.loss
loss.backward()
```

Do note that when training Idefics2 on multi-turn conversations between a user and an assistant, one typically also sets all the tokens corresponding to the user messages to -100.

## Model optimizations: Flash Attention

The code snippets above showcase inference without any optimization tricks. However, one can drastically speed up the model by leveraging [Flash Attention](../perf_train_gpu_one#flash-attention-2), which is a faster implementation of the attention mechanism used inside the model.

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.


```
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of the [flash attention repository](https://github.com/Dao-AILab/flash-attention). Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention-2, simply change the code snippet above with the following change:


```
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    dtype=torch.float16,
+    attn_implementation="flash_attention_2",
).to(device)
```

## Shrinking down Idefics2 using quantization

As the Idefics2 model has 8 billion parameters, that would require about 16GB of GPU RAM in half precision (float16), since each parameter is stored in 2 bytes. However, one can shrink down the size of the model using [quantization](../quantization). If the model is quantized to 4 bits (or half a byte per parameter), that requires only about 3.5GB of RAM.

Quantizing a model is as simple as passing a `quantization_config` to the model. One can change the code snippet above with the changes below. We’ll leverage the BitsAndyBytes quantization (but refer to [this page](../quantization) for other quantization methods):


```
+ from transformers import BitsAndBytesConfig

+ quantization_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_use_double_quant=True,
+    bnb_4bit_compute_dtype=torch.float16
+ )
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
+    dtype=torch.float16,
+    quantization_config=quantization_config,
).to(device)
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Idefics2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

* A notebook on how to fine-tune Idefics2 on a custom dataset using the [Trainer](../main_classes/trainer) can be found [here](https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB?usp=sharing). It supports both full fine-tuning as well as (quantized) LoRa.
* A script regarding how to fine-tune Idefics2 using the TRL library can be found [here](https://gist.github.com/edbeeching/228652fc6c2b29a1641be5a5778223cb).
* Demo notebook regarding fine-tuning Idefics2 for JSON extraction use cases can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Idefics2). 🌎

## Idefics2Config

### class transformers.Idefics2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/configuration_idefics2.py#L171)

( use\_cache = True image\_token\_id = 32001 tie\_word\_embeddings = False vision\_config = None perceiver\_config = None text\_config = None \*\*kwargs  )

Parameters

* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should cache the key/value pairs of the attention mechanism.
* **image\_token\_id** (`int`, *optional*, defaults to 32001) —
  The id of the “image” token.
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether or not to tie the word embeddings with the token embeddings.
* **vision\_config** (`IdeficsVisionConfig` or `dict`, *optional*) —
  Custom vision config or dict
* **perceiver\_config** (`IdeficsPerceiverConfig` or `dict`, *optional*) —
  Custom perceiver config or dict
* **text\_config** (`MistralConfig` or `dict`, *optional*) —
  Custom text config or dict for the text model

This is the configuration class to store the configuration of a [Idefics2Model](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Model). It is used to instantiate a
Idefics2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the model of the Idefics2
[HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import Idefics2Model, Idefics2Config
>>> # Initializing configuration
>>> configuration = Idefics2Config()
>>> # Initializing a model from the configuration
>>> model = Idefics2Model(configuration)
>>> # Accessing the model configuration
>>> configuration = model.config
```

## Idefics2Model

### class transformers.Idefics2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/modeling_idefics2.py#L877)

( config: Idefics2Config  )

Parameters

* **config** ([Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Idefics2 model consisting of a SIGLIP vision encoder and Mistral language decoder

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/modeling_idefics2.py#L1004)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.BoolTensor] = None image\_hidden\_states: typing.Optional[torch.FloatTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None return\_dict: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.modeling\_flash\_attention\_utils.FlashAttentionKwargs]  ) → `transformers.models.idefics2.modeling_idefics2.Idefics2BaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor). See [Idefics2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Idefics2Processor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Processor) uses
  [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **image\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The hidden states of the image encoder after modality projection and perceiver resampling.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.idefics2.modeling_idefics2.Idefics2BaseModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.idefics2.modeling_idefics2.Idefics2BaseModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
  `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder, and optionally by the perceiver

Inputs fed to the model can have an arbitrary number of images. To account for this, pixel\_values fed to
the model have image padding -> (batch\_size, max\_num\_images, 3, max\_heights, max\_widths) where
max\_num\_images is the maximum number of images among the batch\_size samples in the batch.

Padding images are not needed beyond padding the pixel\_values at the entrance of the model.
For efficiency, we only pass through the vision\_model’s forward the real images by
discarding the padding images i.e. pixel\_values of size (image\_batch\_size, 3, height, width) where
image\_batch\_size would be 7 when num\_images\_per\_sample=[1, 3, 1, 2] and max\_num\_images would be 3.

## Idefics2ForConditionalGeneration

### class transformers.Idefics2ForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/modeling_idefics2.py#L1115)

( config  )

Parameters

* **config** ([Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Idefics2 Model with a language modeling head. It is made up a SigLIP vision encoder, with a language modeling head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/modeling_idefics2.py#L1156)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None pixel\_values: typing.Optional[torch.FloatTensor] = None pixel\_attention\_mask: typing.Optional[torch.BoolTensor] = None image\_hidden\_states: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.models.idefics2.modeling_idefics2.Idefics2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **past\_key\_values** (`~cache_utils.Cache`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) —
  The tensors corresponding to the input images. Pixel values can be obtained using
  [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor). See [Idefics2ImageProcessor.**call**()](/docs/transformers/v4.56.2/en/model_doc/fuyu#transformers.FuyuImageProcessor.__call__) for details ([Idefics2Processor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Processor) uses
  [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor) for processing images).
* **pixel\_attention\_mask** (`torch.Tensor` of shape `(batch_size, image_size, image_size)`, *optional*) —
  Mask to avoid performing attention on padding pixel indices.
* **image\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`) —
  The hidden states of the image encoder after modality projection and perceiver resampling.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or `model.image_token_id` (where `model` is your instance of `Idefics2ForConditionalGeneration`).
  Tokens with indices set to `model.image_token_id` are ignored (masked), the loss is only
  computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **logits\_to\_keep** (`Union[int, torch.Tensor]`, defaults to `0`) —
  If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
  `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
  token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
  If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
  This is useful when using packed tensor format (single dimension for batch and sequence length).

Returns

`transformers.models.idefics2.modeling_idefics2.Idefics2CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.idefics2.modeling_idefics2.Idefics2CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Idefics2Config](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2Config)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
  `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
* **image\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*) — Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images, sequence_length, hidden_size)`.
  image\_hidden\_states of the model produced by the vision encoder, and optionally by the perceiver

The [Idefics2ForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from io import BytesIO

>>> from transformers import AutoProcessor, AutoModelForVision2Seq
>>> from transformers.image_utils import load_image

>>> # Note that passing the image urls (instead of the actual pil images) to the processor is also possible
>>> image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
>>> image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
>>> image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")

>>> processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
>>> model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b-base", device_map="auto")

>>> BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
>>> EOS_WORDS_IDS = [processor.tokenizer.eos_token_id]

>>> # Create inputs
>>> prompts = [
...   "<image>In this image, we can see the city of New York, and more specifically the Statue of Liberty.<image>In this image,",
...   "In which city is that bridge located?<image>",
... ]
>>> images = [[image1, image2], [image3]]
>>> inputs = processor(images=images, text=prompts, padding=True, return_tensors="pt").to("cuda")

>>> # Generate
>>> generated_ids = model.generate(**inputs, bad_words_ids=BAD_WORDS_IDS, max_new_tokens=20)
>>> generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

>>> print(generated_texts)
['In this image, we can see the city of New York, and more specifically the Statue of Liberty. In this image, we can see the city of New York, and more specifically the Statue of Liberty.\n\n', 'In which city is that bridge located?\n\nThe bridge is located in the city of Pittsburgh, Pennsylvania.\n\n\nThe bridge is']
```

## Idefics2ImageProcessor

### class transformers.Idefics2ImageProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/image_processing_idefics2.py#L150)

( do\_convert\_rgb: bool = True do\_resize: bool = True size: typing.Optional[dict[str, int]] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: float = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: bool = True do\_image\_splitting: bool = False \*\*kwargs  )

Parameters

* **do\_convert\_rgb** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the image to RGB. This is useful if the input image is of a different format e.g. RGBA.
  Only has an effect if the input image is in the PIL format.
* **do\_resize** (`bool`, *optional*, defaults to `True`) —
  Whether to resize the image. The longest edge of the image is resized to be <= `size["longest_edge"]`, with the
  shortest edge resized to keep the input aspect ratio, with a minimum size of `size["shortest_edge"]`.
* **size** (`Dict`, *optional*) —
  Controls the size of the output image. This is a dictionary containing the keys “shortest\_edge” and “longest\_edge”.
* **resample** (`Resampling`, *optional*, defaults to `Resampling.BILINEAR`) —
  Resampling filter to use when resizing the image.
* **do\_rescale** (`bool`, *optional*, defaults to `True`) —
  Whether to rescale the image. If set to `True`, the image is rescaled to have pixel values between 0 and 1.
* **rescale\_factor** (`float`, *optional*, defaults to `1/255`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the image. If set to `True`, the image is normalized to have a mean of `image_mean` and
  a standard deviation of `image_std`.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_MEAN`) —
  Mean to use if normalizing the image. This is a float or list of floats the length of the number of
  channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
  overridden by the `image_mean` parameter in the `preprocess` method.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `IDEFICS_STANDARD_STD`) —
  Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
  number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
  Can be overridden by the `image_std` parameter in the `preprocess` method.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether or not to pad the images to the largest height and width in the batch and number of images per
  sample in the batch, such that the returned tensor is of shape (batch\_size, max\_num\_images, num\_channels, max\_height, max\_width).
* **do\_image\_splitting** (`bool`, *optional*, defaults to `False`) —
  Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
  strategy was first introduced in <https://huggingface.co/papers/2311.06607>.

Constructs a Idefics image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/image_processing_idefics2.py#L394)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] do\_convert\_rgb: typing.Optional[bool] = None do\_resize: typing.Optional[bool] = None size: typing.Optional[dict[str, int]] = None resample: Resampling = None do\_rescale: typing.Optional[bool] = None rescale\_factor: typing.Optional[float] = None do\_normalize: typing.Optional[bool] = None image\_mean: typing.Union[float, list[float], NoneType] = None image\_std: typing.Union[float, list[float], NoneType] = None do\_pad: typing.Optional[bool] = None do\_image\_splitting: typing.Optional[bool] = None return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None input\_data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = None data\_format: typing.Optional[transformers.image\_utils.ChannelDimension] = <ChannelDimension.FIRST: 'channels\_first'>  )

Parameters

* **images** (`ImageInput`) —
  A list of images to preprocess.
* **do\_convert\_rgb** (`bool`, *optional*, defaults to `self.do_convert_rgb`) —
  Whether to convert the image to RGB.
* **do\_resize** (`bool`, *optional*, defaults to `self.do_resize`) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*, defaults to `self.size`) —
  Size of the image after resizing. Shortest edge of the image is resized to size[“shortest\_edge”], with
  the longest edge resized to keep the input aspect ratio.
* **resample** (`int`, *optional*, defaults to `self.resample`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_rescale** (`bool`, *optional*, defaults to `self.do_rescale`) —
  Whether to rescale the image.
* **rescale\_factor** (`float`, *optional*, defaults to `self.rescale_factor`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*, defaults to `self.do_normalize`) —
  Whether to normalize the image.
* **image\_mean** (`float` or `list[float]`, *optional*, defaults to `self.image_mean`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`float` or `list[float]`, *optional*, defaults to `self.image_std`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_pad** (`bool`, *optional*, defaults to `self.do_pad`) —
  Whether or not to pad the images to the largest height and width in the batch.
* **do\_image\_splitting** (`bool`, *optional*, defaults to `self.do_image_splitting`) —
  Whether to split the image into a sequence 4 equal sub-images concatenated with the original image. That
  strategy was first introduced in <https://huggingface.co/papers/2311.06607>.
* **return\_tensors** (`str` or `TensorType`, *optional*) —
  The type of tensors to return. Can be one of:
  + Unset: Return a list of `np.ndarray`.
  + `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
  + `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
  + `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
  + `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
* **data\_format** (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`) —
  The channel dimension format for the output image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + Unset: Use the channel dimension format of the input image.
* **input\_data\_format** (`ChannelDimension` or `str`, *optional*) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

Preprocess a batch of images.

## Idefics2ImageProcessorFast

### class transformers.Idefics2ImageProcessorFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/image_processing_idefics2_fast.py#L121)

( \*\*kwargs: typing\_extensions.Unpack[transformers.image\_processing\_utils\_fast.DefaultFastImageProcessorKwargs]  )

Constructs a fast Idefics2 image processor.

#### preprocess

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/image_processing_idefics2_fast.py#L219)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']] \*\*kwargs: typing\_extensions.Unpack[transformers.models.idefics2.image\_processing\_idefics2\_fast.Idefics2FastImageProcessorKwargs]  ) → `<class 'transformers.image_processing_base.BatchFeature'>`

Parameters

* **images** (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]`) —
  Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
  passing in images with pixel values between 0 and 1, set `do_rescale=False`.
* **do\_resize** (`bool`, *optional*) —
  Whether to resize the image.
* **size** (`dict[str, int]`, *optional*) —
  Describes the maximum input dimensions to the model.
* **default\_to\_square** (`bool`, *optional*) —
  Whether to default to a square image when resizing, if size is an int.
* **resample** (`Union[PILImageResampling, F.InterpolationMode, NoneType]`) —
  Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
  has an effect if `do_resize` is set to `True`.
* **do\_center\_crop** (`bool`, *optional*) —
  Whether to center crop the image.
* **crop\_size** (`dict[str, int]`, *optional*) —
  Size of the output image after applying `center_crop`.
* **do\_rescale** (`bool`, *optional*) —
  Whether to rescale the image.
* **rescale\_factor** (`Union[int, float, NoneType]`) —
  Rescale factor to rescale the image by if `do_rescale` is set to `True`.
* **do\_normalize** (`bool`, *optional*) —
  Whether to normalize the image.
* **image\_mean** (`Union[float, list[float], NoneType]`) —
  Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
* **image\_std** (`Union[float, list[float], NoneType]`) —
  Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
  `True`.
* **do\_convert\_rgb** (`bool`, *optional*) —
  Whether to convert the image to RGB.
* **return\_tensors** (`Union[str, ~utils.generic.TensorType, NoneType]`) —
  Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
* **data\_format** (`~image_utils.ChannelDimension`, *optional*) —
  Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
* **input\_data\_format** (`Union[str, ~image_utils.ChannelDimension, NoneType]`) —
  The channel dimension format for the input image. If unset, the channel dimension format is inferred
  from the input image. Can be one of:
  + `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
  + `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
  + `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
* **device** (`torch.device`, *optional*) —
  The device to process the images on. If unset, the device is inferred from the input images.
* **disable\_grouping** (`bool`, *optional*) —
  Whether to disable grouping of images by size to process them individually and not in batches.
  If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
  empirical observations, as detailed here: <https://github.com/huggingface/transformers/pull/38157>
* **do\_image\_splitting** (`bool`, *optional*, defaults to `False`) —
  Whether to split the image into a sequence 4 equal sub-images concatenated with the original image.
* **do\_pad** (`bool`, *optional*, defaults to `True`) —
  Whether to pad images to the largest height and width in the batch.

Returns

`<class 'transformers.image_processing_base.BatchFeature'>`

* **data** (`dict`) — Dictionary of lists/arrays/tensors returned by the **call** method (‘pixel\_values’, etc.).
* **tensor\_type** (`Union[None, str, TensorType]`, *optional*) — You can give a tensor\_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
  initialization.

## Idefics2Processor

### class transformers.Idefics2Processor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/processing_idefics2.py#L66)

( image\_processor tokenizer = None image\_seq\_len: int = 64 chat\_template: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **image\_processor** (`Idefics2ImageProcessor`) —
  An instance of [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor). The image processor is a required input.
* **tokenizer** (`PreTrainedTokenizerBase`, *optional*) —
  An instance of [PreTrainedTokenizerBase](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase). This should correspond with the model’s text model. The tokenizer is a required input.
* **image\_seq\_len** (`int`, *optional*, defaults to 64) —
  The length of the image sequence i.e. the number of  tokens per image in the input.
  This parameter is used to build the string from the input prompt and image tokens and should match the
  config.perceiver\_config.resampler\_n\_latents value for the model used.
* **chat\_template** (`str`, *optional*) — A Jinja template which will be used to convert lists of messages
  in a chat into a tokenizable string.

Constructs a IDEFICS2 processor which wraps a LLama tokenizer and IDEFICS2 image processor into a single processor.

[IdeficsProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor) offers all the functionalities of [Idefics2ImageProcessor](/docs/transformers/v4.56.2/en/model_doc/idefics2#transformers.Idefics2ImageProcessor) and [LlamaTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/llama#transformers.LlamaTokenizerFast). See
the docstring of [**call**()](/docs/transformers/v4.56.2/en/model_doc/idefics#transformers.IdeficsProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/idefics2/processing_idefics2.py#L127)

( images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]], list[list[typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']]]]] = None text: typing.Union[str, ForwardRef('PreTokenizedInput'), list[str], list['PreTokenizedInput']] = None audio = None videos = None \*\*kwargs: typing\_extensions.Unpack[transformers.models.idefics2.processing\_idefics2.Idefics2ProcessorKwargs]  )

Parameters

* **images** (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`, *optional*) —
  The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
  tensor. If is of type `list[ImageInput]`, it’s assumed that this is for a single prompt i.e. of batch size 1.
* **text** (`Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

  Wherever an image token, `<image>` is encountered it is expanded to
  `<fake_token_around_image>` + `<image>`  *`image_seq_len`*  `.
* **return\_tensors** (`Union[str, TensorType]`, *optional*) —
  If set, will return tensors of a particular framework. See [PreTrainedTokenizerFast.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for more
  information.

Processes the input prompts and returns a BatchEncoding.

Example:


```
>>> import requests
>>> from transformers import Idefics2Processor
>>> from transformers.image_utils import load_image

>>> processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b", image_seq_len=2)
>>> processor.image_processor.do_image_splitting = False  # Force as False to simplify the example

>>> url1 = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
>>> url2 = "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"

>>> image1, image2 = load_image(url1), load_image(url2)
>>> images = [[image1], [image2]]

>>> text = [
...     "<image>In this image, we see",
...     "bla bla bla<image>",
... ]
>>> outputs = processor(images=images, text=text, return_tensors="pt", padding=True)
>>> input_ids = outputs.input_ids
>>> input_tokens = processor.tokenizer.batch_decode(input_ids)
>>> print(input_tokens)
['<s><fake_token_around_image><image><image><fake_token_around_image> In this image, we see', '<s> bla bla bla<fake_token_around_image><image><image><fake_token_around_image>']
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/idefics2.md)
