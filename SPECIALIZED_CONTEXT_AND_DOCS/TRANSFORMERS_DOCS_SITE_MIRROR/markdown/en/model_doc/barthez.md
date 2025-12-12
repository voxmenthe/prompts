# BARThez

[BARThez](https://huggingface.co/papers/2010.12321) is a [BART](./bart) model designed for French language tasks. Unlike existing French BERT models, BARThez includes a pretrained encoder-decoder, allowing it to generate text as well. This model is also available as a multilingual variant, mBARThez, by continuing pretraining multilingual BART on a French corpus.

You can find all of the original BARThez checkpoints under the [BARThez](https://huggingface.co/collections/dascim/barthez-670920b569a07aa53e3b6887) collection.

> [!TIP]
> This model was contributed by [moussakam](https://huggingface.co/moussakam).
> Refer to the [BART](./bart) docs for more usage examples.

The example below demonstrates how to predict the `` token with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="moussaKam/barthez",
    dtype=torch.float16,
    device=0
)
pipeline("Les plantes produisent  grâce à un processus appelé photosynthèse.")
```

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "moussaKam/barthez",
)
model = AutoModelForMaskedLM.from_pretrained(
    "moussaKam/barthez",
    dtype=torch.float16,
    device_map="auto",
)
inputs = tokenizer("Les plantes produisent  grâce à un processus appelé photosynthèse.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

```bash
echo -e "Les plantes produisent  grâce à un processus appelé photosynthèse." | transformers run --task fill-mask --model moussaKam/barthez --device 0
```

## BarthezTokenizer[[transformers.BarthezTokenizer]]

#### transformers.BarthezTokenizer[[transformers.BarthezTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/barthez/tokenization_barthez.py#L35)

Adapted from [CamembertTokenizer](/docs/transformers/main/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer). Construct a "fast" BARThez tokenizer. Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

**Parameters:**

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.

## BarthezTokenizerFast[[transformers.BarthezTokenizer]]

#### transformers.BarthezTokenizer[[transformers.BarthezTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/barthez/tokenization_barthez.py#L35)

Adapted from [CamembertTokenizer](/docs/transformers/main/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/main/en/model_doc/longformer#transformers.RobertaTokenizer). Construct a "fast" BARThez tokenizer. Based on
[SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

**Parameters:**

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

vocab_file (`str`, *optional*) : [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that contains the vocabulary necessary to instantiate a tokenizer.

vocab (`str`, `dict` or `list`, *optional*) : Custom vocabulary dictionary. If not provided, vocabulary is loaded from vocab_file.

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word.
