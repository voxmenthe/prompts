# BARTpho

[BARTpho](https://huggingface.co/papers/2109.09701) is a large-scale Vietnamese sequence-to-sequence model. It offers a word-based and syllable-based version. This model is built on the [BART](./bart) large architecture with its denoising pretraining.

You can find all the original checkpoints under the [VinAI](https://huggingface.co/vinai/models?search=bartpho) organization.

> [!TIP]
> This model was contributed by [dqnguyen](https://huggingface.co/dqnguyen).
> Check out the right sidebar for examples of how to apply BARTpho to different language tasks.

The example below demonstrates how to summarize text with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
import torch
from transformers import pipeline

pipeline = pipeline(
   task="summarization",
   model="vinai/bartpho-word",
   dtype=torch.float16,
   device=0
)

text = """
Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ
"""
pipeline(text)
```

```python
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "vinai/bartpho-word",
)
model = BartForConditionalGeneration.from_pretrained(
    "vinai/bartpho-word",
    dtype=torch.float16,
    device_map="auto",
)

text = """
Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ
"""
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

```bash
echo -e "Quang tổng hợp hay gọi tắt là quang hợp là quá trình thu nhận và chuyển hóa năng lượng ánh sáng Mặt trời của thực vật,
tảo và một số vi khuẩn để tạo ra hợp chất hữu cơ phục vụ bản thân cũng như làm nguồn thức ăn cho hầu hết các sinh vật
trên Trái Đất. Quang hợp trong thực vật thường liên quan đến chất tố diệp lục màu xanh lá cây và tạo ra oxy như một sản phẩm phụ" | \
transformers run --task summarization --model vinai/bartpho-word --device 0
```

## Notes

- BARTpho uses the large architecture of BART with an additional layer-normalization layer on top of the encoder and decoder. The BART-specific classes should be replaced with the mBART-specific classes.
- This implementation only handles tokenization through the `monolingual_vocab_file` file. This is a Vietnamese-specific subset of token types taken from that multilingual vocabulary. If you want to use this tokenizer for another language, replace the `monolingual_vocab_file` with one specialized for your target language.

## BartphoTokenizer[[transformers.BartphoTokenizer]]

#### transformers.BartphoTokenizer[[transformers.BartphoTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bartpho/tokenization_bartpho.py#L33)

Adapted from [XLMRobertaTokenizer](/docs/transformers/main/en/model_doc/xlm-roberta#transformers.XLMRobertaTokenizer). Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

build_inputs_with_special_tokenstransformers.BartphoTokenizer.build_inputs_with_special_tokenshttps://github.com/huggingface/transformers/blob/main/src/transformers/models/bartpho/tokenization_bartpho.py#L161[{"name": "token_ids_0", "val": ": list"}, {"name": "token_ids_1", "val": ": typing.Optional[list[int]] = None"}]- **token_ids_0** (`list[int]`) --
  List of IDs to which the special tokens will be added.
- **token_ids_1** (`list[int]`, *optional*) --
  Optional second list of IDs for sequence pairs.0`list[int]`List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An BARTPho sequence has the following format:

- single sequence: ` X `
- pair of sequences: ` A  B `

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file. This vocabulary is the pre-trained SentencePiece model available from the multilingual XLM-RoBERTa, also used in mBART, consisting of 250K types.

monolingual_vocab_file (`str`) : Path to the monolingual vocabulary file. This monolingual vocabulary consists of Vietnamese-specialized types extracted from the multilingual vocabulary vocab_file of 250K types.

bos_token (`str`, *optional*, defaults to `""`) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.   

eos_token (`str`, *optional*, defaults to `""`) : The end of sequence token.    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.   

sep_token (`str`, *optional*, defaults to `""`) : The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.

cls_token (`str`, *optional*, defaults to `""`) : The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.

unk_token (`str`, *optional*, defaults to `""`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

pad_token (`str`, *optional*, defaults to `""`) : The token used for padding, for example when batching sequences of different lengths.

mask_token (`str`, *optional*, defaults to `""`) : The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.

sp_model_kwargs (`dict`, *optional*) : Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:  - `enable_sampling`: Enable subword regularization. - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.  - `nbest_size = {0,1}`: No sampling is performed. - `nbest_size > 1`: samples from the nbest_size results. - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.  - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.

sp_model (`SentencePieceProcessor`) : The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).

**Returns:**

``list[int]``

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
#### create_token_type_ids_from_sequences[[transformers.BartphoTokenizer.create_token_type_ids_from_sequences]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bartpho/tokenization_bartpho.py#L215)

Create a mask from the two sequences passed to be used in a sequence-pair classification task. BARTPho does not
make use of token type ids, therefore a list of zeros is returned.

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

**Returns:**

``list[int]``

List of zeros.
#### get_special_tokens_mask[[transformers.BartphoTokenizer.get_special_tokens_mask]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bartpho/tokenization_bartpho.py#L187)

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

**Parameters:**

token_ids_0 (`list[int]`) : List of IDs.

token_ids_1 (`list[int]`, *optional*) : Optional second list of IDs for sequence pairs.

already_has_special_tokens (`bool`, *optional*, defaults to `False`) : Whether or not the token list is already formatted with special tokens for the model.

**Returns:**

``list[int]``

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
#### get_vocab[[transformers.BartphoTokenizer.get_vocab]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bartpho/tokenization_bartpho.py#L245)

Override to use fairseq vocabulary
