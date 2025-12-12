*This model was released on 2024-12-27 and added to Hugging Face Transformers on 2025-03-28.*

# DeepSeek-V3

## Overview

The DeepSeek-V3 model was proposed in [DeepSeek-V3 Technical Report](https://huggingface.co/papers/2412.19437) by DeepSeek-AI Team.

The abstract from the paper is the following:
We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at <https://github.com/deepseek-ai/DeepSeek-V3>.

## Limitations and call for contribution!

We are super happy to make this code community-powered, and would love to see how you can best optimize the following:

* current implementation uses the “naive” attention compution (so not really MLA)
* current implementation loops through the experts. This should be replaced. Pointers to use `get_packed_weights` from `integrations/tensor_parallel`.
* current implementation uses the eleuther formula for ROPE, using the original one would be more efficient! (should still follow our API)
* static cache is not supported (this should be just a generation config issue / config shape issues)

### Usage tips

The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

You can run the model in `FP8` automatically, using 2 nodes of 8 H100 should be more than enough!


```
# `run_deepseek_v1.py`
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

tokenizer = AutoTokenizer.from_pretrained("deepseek-r1")

chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


model = AutoModelForCausalLM.from_pretrained("deepseek-r1", device_map="auto", dtype=torch.bfloat16)
inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
import time
start = time.time()
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.batch_decode(outputs))
print(time.time()-start)
```

This generated:


```
<｜Assistant｜><think>
Okay, the user wants to demonstrate how chat templating works. Let me break down what that means. Chat templating is about structuring the conversation data, especially for models that need specific input formats. Maybe they're referring to something like how messages are formatted with roles (user, assistant, system) in APIs like OpenAI.

First, I should explain what chat templating is. It's the process of formatting conversation data into a structured format that the model can understand. This usually includes roles and content. For example, user messages, assistant responses, and system messages each have their own role tags.

They might want an example. Let me think of a simple conversation. The user says "Hello, how are you?" and the assistant responds "I'm doing great. How can I help you today?" Then the user follows up with wanting to show off chat templating. So the example should include the history and the new message.

In some frameworks, like Hugging Face's Transformers, chat templates are applied using Jinja2 templates. The template might look something like combining system messages, then looping through user and assistant messages with appropriate tags. For instance, using {% for message in messages %} and assigning roles like <|user|>, <|assistant|>, etc.

I should structure the example with the messages array, showing each role and content. Then apply a hypothetical template to convert that into a formatted string the model uses. Also, mention that different models have different templating requirements, like using special tokens or varying role labels.

Wait, the user mentioned "chat templating" in the context of showing off. Maybe they want a practical example they can present. So providing a code snippet or a structured data example would be helpful. Let me outline a typical messages array and then the templated output.

Also, it's important to note that proper templating ensures the model knows the conversation flow, which is crucial for generating coherent responses. Maybe include a note about why it's important, like maintaining context and role-specific processing.

Let me check if there are any common mistakes or things to avoid. For example, not closing tags properly, or mismatching roles. But maybe that's too detailed unless the user asks. Focus on the positive example first.

Putting it all together, the response should have an example messages array, the applied template, and the final formatted string. Maybe use angle brackets or special tokens as placeholders. Also, mention that this helps in training or fine-tuning models with structured data.

I think that's a solid approach. Let me structure it step by step to make it clear.
</think>

Chat templating is a way to structure conversation data (e.g., user/assistant interactions) into a format that language models understand. This is especially important for models trained to handle multi-turn dialogues, where the input must explicitly separate roles (user, assistant, system, etc.) and messages. Let’s break this down with an example!

---

### **Step 1: Raw Conversation History**
Suppose we have this conversation:
- **User**: "Hello, how are you?"
- **Assistant**: "I'm doing great. How can I help you today?"
- **User**: "I'd like to show off how chat templating works!"

---

### **Step 2: Structured Messages**
In frameworks like Hugging Face Transformers or OpenAI, conversations are often formatted as a list of dictionaries with `role` and `content`:
```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

---

### **Step 3: Apply a Chat Template**
A **chat template** converts this structured data into a single string formatted for the model. For example, using a Jinja-style template (common in Hugging Face):

```jinja
{% for message in messages %}
    {% if message['role'] == 'user' %}
        <|user|>{{ message['content'] }}<|end|>
    {% elif message['role'] == 'assistant' %}
        <|assistant|>{{ message['content'] }}<|end|>
    {% endif %}
{% endfor %}
<|assistant|>
```

---

### **Step 4: Final Templated Output**
Applying the template to our `messages` list would produce:
```text
<|user|>Hello, how are you?<|end|>
<|assistant|>I'm doing great. How can I help you today?<|end|>
<|user|>I'd like to show off how chat templating works!<|end|>
<|assistant|>
```

This tells the model:  
1. The conversation history (user/assistant turns).  
2. The model’s turn to generate a response (`<|assistant|>` at the end).  

---

### **Key Notes**:
- **Role Separation**: Tags like `<|user|>` and `<|assistant|>` help the model distinguish speakers.
- **Special Tokens**: Models often use unique tokens (e.g., `<|end|>`) to mark message boundaries.
- **Flexibility**: Templates vary by model (e.g., OpenAI uses `{"role": "user", "content": "..."}` instead of tags).

---

### **Why This Matters**:
- **Consistency**: Ensures the model understands dialogue structure.
- **Context Preservation**: Maintains the flow of multi-turn conversations.
- **Alignment**: Matches the format the model was trained on for better performance.

Want to dive deeper or see a specific framework’s implementation (e.g., OpenAI, Llama, Mistral)? Let me know! 😊<｜end▁of▁sentence｜>
```

Use the following to run it


```
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0|1 --rdzv-id an_id --rdzv-backend c10d --rdzv-endpoint master_addr:master_port run_deepseek_r1.py
```

If you have:


```
[rank0]: ncclInternalError: Internal check failed.
[rank0]: Last error:
[rank0]: Bootstrap : no socket interface found
```

error, it means NCCL was probably not loaded.

## DeepseekV3Config

### class transformers.DeepseekV3Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py#L26)

( vocab\_size = 129280 hidden\_size = 7168 intermediate\_size = 18432 moe\_intermediate\_size = 2048 num\_hidden\_layers = 61 num\_attention\_heads = 128 num\_key\_value\_heads = 128 n\_shared\_experts = 1 n\_routed\_experts = 256 routed\_scaling\_factor = 2.5 kv\_lora\_rank = 512 q\_lora\_rank = 1536 qk\_rope\_head\_dim = 64 v\_head\_dim = 128 qk\_nope\_head\_dim = 128 n\_group = 8 topk\_group = 4 num\_experts\_per\_tok = 8 first\_k\_dense\_replace = 3 norm\_topk\_prob = True hidden\_act = 'silu' max\_position\_embeddings = 4096 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = None bos\_token\_id = 0 eos\_token\_id = 1 pretraining\_tp = 1 tie\_word\_embeddings = False rope\_theta = 10000.0 rope\_scaling = None rope\_interleave = True attention\_bias = False attention\_dropout = 0.0 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 129280) —
  Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed when calling [DeepseekV3Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Model)
* **hidden\_size** (`int`, *optional*, defaults to 7168) —
  Dimension of the hidden representations.
* **intermediate\_size** (`int`, *optional*, defaults to 18432) —
  Dimension of the MLP representations.
* **moe\_intermediate\_size** (`int`, *optional*, defaults to 2048) —
  Dimension of the MoE representations.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 61) —
  Number of hidden layers in the Transformer decoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 128) —
  Number of attention heads for each attention layer in the Transformer decoder.
* **num\_key\_value\_heads** (`int`, *optional*, defaults to 128) —
  This is the number of key\_value heads that should be used to implement Grouped Query Attention. If
  `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
  `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to` num\_attention\_heads`.
* **n\_shared\_experts** (`int`, *optional*, defaults to 1) —
  Number of shared experts.
* **n\_routed\_experts** (`int`, *optional*, defaults to 256) —
  Number of routed experts.
* **routed\_scaling\_factor** (`float`, *optional*, defaults to 2.5) —
  Scaling factor or routed experts.
* **kv\_lora\_rank** (`int`, *optional*, defaults to 512) —
  Rank of the LoRA matrices for key and value projections.
* **q\_lora\_rank** (`int`, *optional*, defaults to 1536) —
  Rank of the LoRA matrices for query projections.
* **qk\_rope\_head\_dim** (`int`, *optional*, defaults to 64) —
  Dimension of the query/key heads that use rotary position embeddings.
* **v\_head\_dim** (`int`, *optional*, defaults to 128) —
  Dimension of the value heads.
* **qk\_nope\_head\_dim** (`int`, *optional*, defaults to 128) —
  Dimension of the query/key heads that don’t use rotary position embeddings.
* **n\_group** (`int`, *optional*, defaults to 8) —
  Number of groups for routed experts.
* **topk\_group** (`int`, *optional*, defaults to 4) —
  Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
* **num\_experts\_per\_tok** (`int`, *optional*, defaults to 8) —
  Number of selected experts, None means dense model.
* **first\_k\_dense\_replace** (`int`, *optional*, defaults to 3) —
  Number of dense layers in shallow layers(embed->dense->dense->…->dense->moe->moe…->lm\_head).
  --k dense layers—/
* **norm\_topk\_prob** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the weights of the routed experts.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"silu"`) —
  The non-linear activation function (function or string) in the decoder.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 4096) —
  The maximum sequence length that this model might ever be used with.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **rms\_norm\_eps** (`float`, *optional*, defaults to 1e-06) —
  The epsilon used by the rms normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models). Only
  relevant if `config.is_decoder=True`.
* **pad\_token\_id** (`int`, *optional*) —
  Padding token id.
* **bos\_token\_id** (`int`, *optional*, defaults to 0) —
  Beginning of stream token id.
* **eos\_token\_id** (`int`, *optional*, defaults to 1) —
  End of stream token id.
* **pretraining\_tp** (`int`, *optional*, defaults to 1) —
  Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
  document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
  necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
  issue](https://github.com/pytorch/pytorch/issues/76232).
* **tie\_word\_embeddings** (`bool`, *optional*, defaults to `False`) —
  Whether to tie weight embeddings
* **rope\_theta** (`float`, *optional*, defaults to 10000.0) —
  The base period of the RoPE embeddings.
* **rope\_scaling** (`Dict`, *optional*) —
  Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
  strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
  `{"type": strategy name, "factor": scaling factor}`. When using this flag, don’t update
  `max_position_embeddings` to the expected new maximum.
* **rope\_interleave** (`bool`, *optional*, defaults to `True`) —
  Whether to interleave the rotary position embeddings.
* **attention\_bias** (`bool`, defaults to `False`, *optional*, defaults to `False`) —
  Whether to use a bias in the query, key, value and output projection layers during self-attention.
* **attention\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the attention probabilities.

This is the configuration class to store the configuration of a [DeepseekV3Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Model). It is used to instantiate an DeepSeek
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the DeepSeek-V3.
e.g. [bzantium/tiny-deepseek-v3](https://huggingface.co/bzantium/tiny-deepseek-v3)
Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import DeepseekV3Model, DeepseekV3Config

>>> # Initializing a Deepseek-V3 style configuration
>>> configuration = DeepseekV3Config()

>>> # Accessing the model configuration
>>> configuration = model.config
```

## DeepseekV3Model

### class transformers.DeepseekV3Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L519)

( config: DeepseekV3Config  )

Parameters

* **config** ([DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Deepseek V3 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L538)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None cache\_position: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **cache\_position** (`torch.LongTensor` of shape `(sequence_length)`, *optional*) —
  Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
  this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
  the complete sequence length.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [DeepseekV3Model](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

## DeepseekV3ForCausalLM

### class transformers.DeepseekV3ForCausalLM

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L600)

( config  )

Parameters

* **config** ([DeepseekV3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForCausalLM)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Deepseek V3 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L614)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None cache\_position: typing.Optional[torch.LongTensor] = None logits\_to\_keep: typing.Union[int, torch.Tensor] = 0 \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).
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

[transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DeepseekV3Config](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3Config)) and inputs.

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

The [DeepseekV3ForCausalLM](/docs/transformers/v4.56.2/en/model_doc/deepseek_v3#transformers.DeepseekV3ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

>>> model = DeepseekV3ForCausalLM.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## DeepseekV3ForSequenceClassification

### class transformers.DeepseekV3ForSequenceClassification

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L675)

( config  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/modeling_layers.py#L111)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None position\_ids: typing.Optional[torch.LongTensor] = None past\_key\_values: typing.Optional[transformers.cache\_utils.Cache] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None \*\*kwargs: typing\_extensions.Unpack[transformers.utils.generic.TransformersKwargs]  ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

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
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
  (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
* **use\_cache** (`bool`, *optional*) —
  If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
  `past_key_values`).

Returns

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **past\_key\_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The `GenericForSequenceClassification` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v3.md)
