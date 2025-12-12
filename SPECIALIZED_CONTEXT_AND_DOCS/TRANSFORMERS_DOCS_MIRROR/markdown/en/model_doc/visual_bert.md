*This model was released on 2019-08-09 and added to Hugging Face Transformers on 2021-06-02.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# VisualBERT

[VisualBERT](https://huggingface.co/papers/1908.03557) is a vision-and-language model. It uses an approach called “early fusion”, where inputs are fed together into a single Transformer stack initialized from [BERT](./bert). Self-attention implicitly aligns words with their corresponding image objects. It processes text with visual features from object-detector regions instead of raw pixels.

You can find all the original VisualBERT checkpoints under the [UCLA NLP](https://huggingface.co/uclanlp/models?search=visualbert) organization.

This model was contributed by [gchhablani](https://huggingface.co/gchhablani).
Click on the VisualBERT models in the right sidebar for more examples of how to apply VisualBERT to different image and language tasks.

The example below demonstrates how to answer a question based on an image with the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

AutoModel


```
import torch
import torchvision
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import requests
from io import BytesIO

def get_visual_embeddings_simple(image, device=None):
    
    model = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        raise ValueError("Image must be a PIL Image or path to image file")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    batch_size = features.shape[0]
    feature_dim = features.shape[1]
    visual_seq_length = 10
    
    visual_embeds = features.squeeze(-1).squeeze(-1).unsqueeze(1).expand(batch_size, visual_seq_length, feature_dim)
    
    return visual_embeds

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

response = requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
image = Image.open(BytesIO(response.content))
    
visual_embeds = get_visual_embeddings_simple(image)
    
inputs = tokenizer("What is shown in this image?", return_tensors="pt")
    
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    
inputs.update({
    "visual_embeds": visual_embeds,
    "visual_token_type_ids": visual_token_type_ids,
    "visual_attention_mask": visual_attention_mask,
})
    
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_answer_idx = logits.argmax(-1).item()

print(f"Predicted answer: {predicted_answer_idx}")
```

## Notes

* Use a fine-tuned checkpoint for downstream tasks, like `visualbert-vqa` for visual question answering. Otherwise, use one of the pretrained checkpoints.
* The fine-tuned detector and weights aren’t provided (available in the research projects), but the states can be directly loaded into the detector.
* The text input is concatenated in front of the visual embeddings in the embedding layer and is expected to be bound by `[CLS]` and `SEP` tokens.
* The segment ids must be set appropriately for the text and visual parts.
* Use [BertTokenizer](/docs/transformers/v4.56.2/en/model_doc/bert#transformers.BertTokenizer) to encode the text and implement a custom detector/image processor to get the visual embeddings.

## Resources

* Refer to this [notebook](https://github.com/huggingface/transformers-research-projects/tree/main/visual_bert) for an example of using VisualBERT for visual question answering.
* Refer to this [notebook](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing) for an example of how to generate visual embeddings.

## VisualBertConfig

### class transformers.VisualBertConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/configuration_visual_bert.py#L24)

( vocab\_size = 30522 hidden\_size = 768 visual\_embedding\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 bypass\_transformer = False special\_visual\_initialize = True pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 30522) —
  Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel). Vocabulary size of the model. Defines the
  different tokens that can be represented by the `inputs_ids` passed to the forward method of
  [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel).
* **hidden\_size** (`int`, *optional*, defaults to 768) —
  Dimensionality of the encoder layers and the pooler layer.
* **visual\_embedding\_dim** (`int`, *optional*, defaults to 512) —
  Dimensionality of the visual embeddings to be passed to the model.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 12) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 12) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **intermediate\_size** (`int`, *optional*, defaults to 3072) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"gelu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
* **attention\_probs\_dropout\_prob** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **max\_position\_embeddings** (`int`, *optional*, defaults to 512) —
  The maximum sequence length that this model might ever be used with. Typically set this to something large
  just in case (e.g., 512 or 1024 or 2048).
* **type\_vocab\_size** (`int`, *optional*, defaults to 2) —
  The vocabulary size of the `token_type_ids` passed when calling [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel).
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-12) —
  The epsilon used by the layer normalization layers.
* **bypass\_transformer** (`bool`, *optional*, defaults to `False`) —
  Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
  model directly concatenates the visual embeddings from `VisualBertEmbeddings` with text output from
  transformers, and then pass it to a self-attention layer.
* **special\_visual\_initialize** (`bool`, *optional*, defaults to `True`) —
  Whether or not the visual token type and position type embedding weights should be initialized the same as
  the textual token type and positive type embeddings. When set to `True`, the weights of the textual token
  type and position type embeddings are copied to the respective visual embedding layers.

This is the configuration class to store the configuration of a [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel). It is used to instantiate an
VisualBERT model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the VisualBERT
[uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import VisualBertConfig, VisualBertModel

>>> # Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
>>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

>>> # Initializing a model (with random weights) from the visualbert-vqa-coco-pre style configuration
>>> model = VisualBertModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VisualBertModel

### class transformers.VisualBertModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L551)

( config add\_pooling\_layer = True  )

Parameters

* **config** ([VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **add\_pooling\_layer** (`bool`, *optional*, defaults to `True`) —
  Whether to add a pooling layer

The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is
all you need](https://huggingface.co/papers/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L587)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None  ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

[transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
* **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertModel](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image.
from transformers import AutoTokenizer, VisualBertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

## VisualBertForPreTraining

### class transformers.VisualBertForPreTraining

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L757)

( config  )

Parameters

* **config** ([VisualBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
`sentence-image prediction (classification)` head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L776)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None sentence\_image\_labels: typing.Optional[torch.LongTensor] = None  ) → `transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput` or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
* **sentence\_image\_labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence pair
  (see `input_ids` docstring) Indices should be in `[0, 1]`:
  + 0 indicates sequence B is a matching pair of sequence A for the given image,
  + 1 indicates sequence B is a random sequence w.r.t A for the given image.

Returns

`transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **loss** (`*optional*`, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the sentence-image prediction
  (classification) loss.
* **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **seq\_relationship\_logits** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
  before SoftMax).
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertForPreTraining](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForPreTraining

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)
max_length = inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]
labels = tokenizer(
    "The capital of France is Paris.", return_tensors="pt", padding="max_length", max_length=max_length
)["input_ids"]
sentence_image_labels = torch.tensor(1).unsqueeze(0)  # Batch_size


outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)
loss = outputs.loss
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits
```

## VisualBertForQuestionAnswering

### class transformers.VisualBertForQuestionAnswering

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1097)

( config  )

Parameters

* **config** ([VisualBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
output) for VQA.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1109)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertForQuestionAnswering](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```

## VisualBertForMultipleChoice

### class transformers.VisualBertForMultipleChoice

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L910)

( config  )

Parameters

* **config** ([VisualBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Visual Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L921)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
  `input_ids` above)

Returns

[transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided) — Classification loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — *num\_choices* is the second dimension of the input tensors. (see *input\_ids* above).

  Classification scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertForMultipleChoice](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

visual_embeds = get_visual_embeddings(image)
# (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)
# batch size is 1
inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
inputs_dict.update(
    {
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "visual_token_type_ids": visual_token_type_ids,
        "labels": labels,
    }
)
outputs = model(**inputs_dict)

loss = outputs.loss
logits = outputs.logits
```

## VisualBertForVisualReasoning

### class transformers.VisualBertForVisualReasoning

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1235)

( config  )

Parameters

* **config** ([VisualBertForVisualReasoning](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForVisualReasoning)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
output) for Visual Reasoning e.g. for NLVR task.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1247)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.LongTensor` of shape `(batch_size,)`, *optional*) —
  Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. A classification loss is computed (Cross-Entropy) against these labels.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertForVisualReasoning](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForVisualReasoning) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForVisualReasoning
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor(1).unsqueeze(0)  # Batch size 1, Num choices 2

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```

## VisualBertForRegionToPhraseAlignment

### class transformers.VisualBertForRegionToPhraseAlignment

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1404)

( config  )

Parameters

* **config** ([VisualBertForRegionToPhraseAlignment](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForRegionToPhraseAlignment)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
e.g. for Flickr30 Entities task.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/visual_bert/modeling_visual_bert.py#L1418)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.LongTensor] = None token\_type\_ids: typing.Optional[torch.LongTensor] = None position\_ids: typing.Optional[torch.LongTensor] = None head\_mask: typing.Optional[torch.LongTensor] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None visual\_embeds: typing.Optional[torch.FloatTensor] = None visual\_attention\_mask: typing.Optional[torch.LongTensor] = None visual\_token\_type\_ids: typing.Optional[torch.LongTensor] = None image\_text\_alignment: typing.Optional[torch.LongTensor] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None region\_to\_phrase\_position: typing.Optional[torch.LongTensor] = None labels: typing.Optional[torch.LongTensor] = None  ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **attention\_mask** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
  + 0 corresponds to a *sentence A* token,
  + 1 corresponds to a *sentence B* token.

  [What are token type IDs?](../glossary#token-type-ids)
* **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

  [What are position IDs?](../glossary#position-ids)
* **head\_mask** (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*) —
  Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
  + 1 indicates the head is **not masked**,
  + 0 indicates the head is **masked**.
* **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **visual\_embeds** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length, visual_embedding_dim)`, *optional*) —
  The embedded representation of the visual inputs, generally derived using using an object detector.
* **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Mask to avoid performing attention on visual embeddings. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **visual\_token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, visual_seq_length)`, *optional*) —
  Segment token indices to indicate different portions of the visual embeds.

  [What are token type IDs?](../glossary#token-type-ids) The authors of VisualBERT set the
  *visual\_token\_type\_ids* to *1* for all tokens.
* **image\_text\_alignment** (`torch.LongTensor` of shape `(batch_size, visual_seq_length, alignment_number)`, *optional*) —
  Image-Text alignment uses to decide the position IDs of the visual embeddings.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **region\_to\_phrase\_position** (`torch.LongTensor` of shape `(batch_size, total_sequence_length)`, *optional*) —
  The positions depicting the position of the image embedding corresponding to the textual tokens.
* **labels** (`torch.LongTensor` of shape `(batch_size, total_sequence_length, visual_sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and the
  outputs from the attention layer.

Returns

[transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VisualBertConfig](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
* **hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VisualBertForRegionToPhraseAlignment](/docs/transformers/v4.56.2/en/model_doc/visual_bert#transformers.VisualBertForRegionToPhraseAlignment) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
# Assumption: *get_visual_embeddings(image)* gets the visual embeddings of the image in the batch.
from transformers import AutoTokenizer, VisualBertForRegionToPhraseAlignment
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = VisualBertForRegionToPhraseAlignment.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]))

inputs.update(
    {
        "region_to_phrase_position": region_to_phrase_position,
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.ones(
    (1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2], visual_embeds.shape[-2])
)  # Batch size 1

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/visual_bert.md)
