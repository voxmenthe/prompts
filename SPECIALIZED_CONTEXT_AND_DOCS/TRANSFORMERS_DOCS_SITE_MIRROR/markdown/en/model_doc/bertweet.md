# BERTweet

    
    

## BERTweet

[BERTweet](https://huggingface.co/papers/2005.10200) shares the same architecture as [BERT-base](./bert), but it's pretrained like [RoBERTa](./roberta) on English Tweets. It performs really well on Tweet-related tasks like part-of-speech tagging, named entity recognition, and text classification.

You can find all the original BERTweet checkpoints under the [VinAI Research](https://huggingface.co/vinai?search_models=BERTweet) organization.

> [!TIP]
> Refer to the [BERT](./bert) docs for more examples of how to apply BERTweet to different language tasks.

The example below demonstrates how to predict the `` token with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline), [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel), and from the command line.

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="vinai/bertweet-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create  through a process known as photosynthesis.")
```

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
   "vinai/bertweet-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "vinai/bertweet-base",
    dtype=torch.float16,
    device_map="auto"
)
inputs = tokenizer("Plants create  through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

```bash
echo -e "Plants create  through a process known as photosynthesis." | transformers run --task fill-mask --model vinai/bertweet-base --device 0
```

## Notes

- Use the [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer) or [BertweetTokenizer](/docs/transformers/main/en/model_doc/bertweet#transformers.BertweetTokenizer) because it's preloaded with a custom vocabulary adapted to tweet-specific tokens like hashtags (#), mentions (@), emojis, and common abbreviations. Make sure to also install the [emoji](https://pypi.org/project/emoji/) library.
- Inputs should be padded on the right (`padding="max_length"`) because BERT uses absolute position embeddings.

## BertweetTokenizer[[transformers.BertweetTokenizer]]

#### transformers.BertweetTokenizer[[transformers.BertweetTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L53)

Constructs a BERTweet tokenizer, using Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

add_from_filetransformers.BertweetTokenizer.add_from_filehttps://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L334[{"name": "f", "val": ""}]

Loads a pre-existing dictionary from a text file and adds its symbols to this instance.

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

merges_file (`str`) : Path to the merges file.

normalization (`bool`, *optional*, defaults to `False`) : Whether or not to apply a normalization preprocess.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
#### convert_tokens_to_string[[transformers.BertweetTokenizer.convert_tokens_to_string]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L293)

Converts a sequence of tokens (string) in a single string.
#### normalizeToken[[transformers.BertweetTokenizer.normalizeToken]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L266)

Normalize tokens in a Tweet
#### normalizeTweet[[transformers.BertweetTokenizer.normalizeTweet]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L232)

Normalize a raw Tweet
#### save_vocabulary[[transformers.BertweetTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bertweet/tokenization_bertweet.py#L304)

Save the vocabulary and merges files to a directory.
