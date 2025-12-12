*This model was released on 2023-08-22 and added to Hugging Face Transformers on 2023-10-23.*

# SeamlessM4T

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The SeamlessM4T model was proposed in [SeamlessM4T — Massively Multilingual & Multimodal Machine Translation](https://huggingface.co/papers/2308.11596) by the Seamless Communication team from Meta AI.

This is the **version 1** release of the model. For the updated **version 2** release, refer to the [Seamless M4T v2 docs](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t_v2).

SeamlessM4T is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.

SeamlessM4T enables multiple tasks without relying on separate models:

* Speech-to-speech translation (S2ST)
* Speech-to-text translation (S2TT)
* Text-to-speech translation (T2ST)
* Text-to-text translation (T2TT)
* Automatic speech recognition (ASR)

[SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) can perform all the above tasks, but each task also has its own dedicated sub-model.

The abstract from the paper is the following:

*What does it take to create the Babel Fish, a tool that can help individuals translate speech between any two languages? While recent breakthroughs in text-based models have pushed machine translation coverage beyond 200 languages, unified speech-to-speech translation models have yet to achieve similar strides. More specifically, conventional speech-to-speech translation systems rely on cascaded systems that perform translation progressively, putting high-performing unified systems out of reach. To address these gaps, we introduce SeamlessM4T, a single model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages. To build this, we used 1 million hours of open speech audio data to learn self-supervised speech representations with w2v-BERT 2.0. Subsequently, we created a multimodal corpus of automatically aligned speech translations. Filtered and combined with human-labeled and pseudo-labeled data, we developed the first multilingual system capable of translating from and into English for both speech and text. On FLEURS, SeamlessM4T sets a new standard for translations into multiple target languages, achieving an improvement of 20% BLEU over the previous SOTA in direct speech-to-text translation. Compared to strong cascaded models, SeamlessM4T improves the quality of into-English translation by 1.3 BLEU points in speech-to-text and by 2.6 ASR-BLEU points in speech-to-speech. Tested for robustness, our system performs better against background noises and speaker variations in speech-to-text tasks compared to the current SOTA model. Critically, we evaluated SeamlessM4T on gender bias and added toxicity to assess translation safety. Finally, all contributions in this work are open-sourced and accessible at <https://github.com/facebookresearch/seamless_communication>*

## Usage

First, load the processor and a checkpoint of the model:


```
>>> from transformers import AutoProcessor, SeamlessM4TModel

>>> processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
>>> model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
```

You can seamlessly use this model on text or on audio, to generated either translated text or translated audio.

Here is how to use the processor to process text and audio:


```
>>> # let's load an audio sample from an Arabic speech corpus
>>> from datasets import load_dataset
>>> dataset = load_dataset("halabi2016/arabic_speech_corpus", split="test", streaming=True)
>>> audio_sample = next(iter(dataset))["audio"]

>>> # now, process it
>>> audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt")

>>> # now, process some English test as well
>>> text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```

### Speech

[SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) can *seamlessly* generate text or speech with few or no changes. Let’s target Russian voice translation:


```
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

With basically the same code, I’ve translated English text and Arabic speech to Russian speech samples.

### Text

Similarly, you can generate translated text from audio files or from text with the same model. You only have to pass `generate_speech=False` to [SeamlessM4TModel.generate()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel.generate).
This time, let’s translate to French.


```
>>> # from audio
>>> output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

>>> # from text
>>> output_tokens = model.generate(**text_inputs, tgt_lang="fra", generate_speech=False)
>>> translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
```

### Tips

#### 1. Use dedicated models

[SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code:


```
>>> from transformers import SeamlessM4TForSpeechToSpeech
>>> model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Or you can replace the text-to-text generation snippet with the model dedicated to the T2TT task, you only have to remove `generate_speech=False`.


```
>>> from transformers import SeamlessM4TForTextToText
>>> model = SeamlessM4TForTextToText.from_pretrained("facebook/hf-seamless-m4t-medium")
```

Feel free to try out [SeamlessM4TForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToText) and [SeamlessM4TForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToSpeech) as well.

#### 2. Change the speaker identity

You have the possibility to change the speaker used for speech synthesis with the `spkr_id` argument. Some `spkr_id` works better than other for some languages!

#### 3. Change the generation strategy

You can use different [generation strategies](./generation_strategies) for speech and text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, speech_do_sample=True)` which will successively perform beam-search decoding on the text model, and multinomial sampling on the speech model.

#### 4. Generate speech and text at the same time

Use `return_intermediate_token_ids=True` with [SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel) to return both speech and text !

## Model architecture

SeamlessM4T features a versatile architecture that smoothly handles the sequential generation of text and speech. This setup comprises two sequence-to-sequence (seq2seq) models. The first model translates the input modality into translated text, while the second model generates speech tokens, known as “unit tokens,” from the translated text.

Each modality has its own dedicated encoder with a unique architecture. Additionally, for speech output, a vocoder inspired by the [HiFi-GAN](https://huggingface.co/papers/2010.05646) architecture is placed on top of the second seq2seq model.

Here’s how the generation process works:

* Input text or speech is processed through its specific encoder.
* A decoder creates text tokens in the desired language.
* If speech generation is required, the second seq2seq model, following a standard encoder-decoder structure, generates unit tokens.
* These unit tokens are then passed through the final vocoder to produce the actual speech.

This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4TModel

### class transformers.SeamlessM4TModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L3631)

( config current\_modality = 'text'  )

Parameters

* **config** ([SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **current\_modality** (`str`, *optional*, defaults to `"text"`) —
  Default modality. Used to initialize the model.

The original SeamlessM4T Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L3841)

( input\_ids: typing.Optional[torch.Tensor] = None input\_features: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None spkr\_id: typing.Optional[int] = 0 generate\_speech: typing.Optional[bool] = True \*\*kwargs  ) → `Union[SeamlessM4TGenerationOutput, tuple[Tensor], ModelOutput]`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SeamlessM4TTokenizer](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizer) or [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`, *optional*) —
  Input audio features. This should be returned by the [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) class or the
  [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) class. See [SeamlessM4TFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) for details.
* **return\_intermediate\_token\_ids** (`bool`, *optional*) —
  If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
  to get translated text alongside the audio. Note that if `generate_speech=True`, this parameter will be
  ignored.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **spkr\_id** (`int`, *optional*, defaults to 0) —
  The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
* **generate\_speech** (`bool`, *optional*, defaults to `True`) —
  If `False`, will only returns the text tokens and won’t generate speech.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to [GenerationMixin.generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate). Keyword
  arguments are of two types:
  + Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
    except for `decoder_input_ids` which will only be passed through the text components.
  + With a *text\_* or *speech\_* prefix, they will be input for the `generate` method of the
    text model and speech model respectively. It has the priority over the keywords without a prefix.

  This means you can, for example, specify a generation strategy for one generation but not for the
  other.

Returns

`Union[SeamlessM4TGenerationOutput, tuple[Tensor], ModelOutput]`

* If `generate_speech` and `return_intermediate_token_ids`, returns `SeamlessM4TGenerationOutput`.
* If `generate_speech` and not `return_intermediate_token_ids`, returns a tuple composed of waveforms of
  shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.
* If `generate_speech=False`, it will returns `ModelOutput`.

Generates translated token ids and/or translated audio waveforms.

This method successively calls the `.generate` function of two different sub-models. You can specify keyword
arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
that will be passed to one of them.

For example, calling `.generate(input_ids=input_ids, num_beams=4, speech_do_sample=True)` will successively
perform beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4TForTextToSpeech

### class transformers.SeamlessM4TForTextToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2977)

( config: SeamlessM4TConfig  )

Parameters

* **config** ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L3127)

( input\_ids: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None spkr\_id: typing.Optional[int] = 0 \*\*kwargs  ) → `Union[SeamlessM4TGenerationOutput, tuple[Tensor]]`

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SeamlessM4TTokenizer](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizer) or [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **return\_intermediate\_token\_ids** (`bool`, *optional*) —
  If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
  to get translated text alongside the audio.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **spkr\_id** (`int`, *optional*, defaults to 0) —
  The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to [GenerationMixin.generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate). Keyword
  arguments are of two types:
  + Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
    except for `decoder_input_ids` which will only be passed through the text components.
  + With a *text\_* or *speech\_* prefix, they will be input for the `generate` method of the
    text model and speech model respectively. It has the priority over the keywords without a prefix.

  This means you can, for example, specify a generation strategy for one generation but not for the
  other.

Returns

`Union[SeamlessM4TGenerationOutput, tuple[Tensor]]`

* If `return_intermediate_token_ids`, returns `SeamlessM4TGenerationOutput`.
* If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.

Generates translated audio waveforms.

This method successively calls the `.generate` function of two different sub-models. You can specify keyword
arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
that will be passed to one of them.

For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4TForSpeechToSpeech

### class transformers.SeamlessM4TForSpeechToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L3300)

( config  )

Parameters

* **config** ([SeamlessM4TForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToSpeech)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The speech-to-speech SeamlessM4T Model transformer which can be used for S2ST.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L3451)

( input\_features: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None spkr\_id: typing.Optional[int] = 0 \*\*kwargs  ) → `Union[SeamlessM4TGenerationOutput, tuple[Tensor]]`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`) —
  Input audio features. This should be returned by the [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) class or the
  [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) class. See [SeamlessM4TFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) for details.
* **return\_intermediate\_token\_ids** (`bool`, *optional*) —
  If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
  to get translated text alongside the audio.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **spkr\_id** (`int`, *optional*, defaults to 0) —
  The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to [GenerationMixin.generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate). Keyword
  arguments are of two types:
  + Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
    except for `decoder_input_ids` which will only be passed through the text components.
  + With a *text\_* or *speech\_* prefix, they will be input for the `generate` method of the
    text model and speech model respectively. It has the priority over the keywords without a prefix.

  This means you can, for example, specify a generation strategy for one generation but not for the
  other.

Returns

`Union[SeamlessM4TGenerationOutput, tuple[Tensor]]`

* If `return_intermediate_token_ids`, returns `SeamlessM4TGenerationOutput`.
* If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.

Generates translated audio waveforms.

This method successively calls the `.generate` function of two different sub-models. You can specify keyword
arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
that will be passed to one of them.

For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform
beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4TForTextToText

### class transformers.SeamlessM4TForTextToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2459)

( config: SeamlessM4TConfig  )

Parameters

* **config** ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text-to-text SeamlessM4T Model transformer which can be used for T2TT.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2501)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

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
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  For translation and summarization training, `decoder_input_ids` should be provided. If no
  `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
  for denoising pre-training following the paper.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_bart._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
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

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [SeamlessM4TForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2599)

( input\_ids = None tgt\_lang = None generation\_config = None logits\_processor = None stopping\_criteria = None prefix\_allowed\_tokens\_fn = None synced\_gpus = False \*\*kwargs  ) → [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Parameters

* **input\_ids** (`torch.Tensor` of varying shape depending on the modality, *optional*) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SeamlessM4TTokenizer](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizer) or [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor). See
  [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **generation\_config** (`~generation.GenerationConfig`, *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which had the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complement the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **synced\_gpus** (`bool`, *optional*, defaults to `False`) —
  Whether to continue running the while loop until max\_length (needed to avoid deadlocking with
  `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model.

Returns

[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
* [GenerateBeamEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4TForSpeechToText

### class transformers.SeamlessM4TForSpeechToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2716)

( config: SeamlessM4TConfig  )

Parameters

* **config** ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The speech-to-text SeamlessM4T Model transformer which can be used for S2TT.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2753)

( input\_features: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None \*\*kwargs  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`) —
  Input audio features. This should be returned by the [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) class or the
  [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) class. See [SeamlessM4TFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) for details.
* **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
  + 1 for tokens that are **not masked**,
  + 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  For translation and summarization training, `decoder_input_ids` should be provided. If no
  `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
  for denoising pre-training following the paper.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_bart._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
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

Returns

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [SeamlessM4TForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, SeamlessM4TForSpeechToText
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained(""facebook/hf-seamless-m4t-medium"")
>>> model = SeamlessM4TForSpeechToText.from_pretrained(""facebook/hf-seamless-m4t-medium"")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2858)

( input\_features = None tgt\_lang = None generation\_config = None logits\_processor = None stopping\_criteria = None prefix\_allowed\_tokens\_fn = None synced\_gpus = False \*\*kwargs  ) → [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`) —
  Input audio features. This should be returned by the [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) class or the
  [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) class. See [SeamlessM4TFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) for details.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **generation\_config** (`~generation.GenerationConfig`, *optional*) —
  The generation configuration to be used as base parametrization for the generation call. `**kwargs`
  passed to generate matching the attributes of `generation_config` will override them. If
  `generation_config` is not provided, the default will be used, which had the following loading
  priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
  configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationConfig)’s
  default values, whose documentation should be checked to parameterize generation.
* **logits\_processor** (`LogitsProcessorList`, *optional*) —
  Custom logits processors that complement the default logits processors built from arguments and
  generation config. If a logit processor is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **stopping\_criteria** (`StoppingCriteriaList`, *optional*) —
  Custom stopping criteria that complement the default stopping criteria built from arguments and a
  generation config. If a stopping criteria is passed that is already created with the arguments or a
  generation config an error is thrown. This feature is intended for advanced users.
* **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], list[int]]`, *optional*) —
  If provided, this function constraints the beam search to allowed tokens only at each step. If not
  provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
  `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
  on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
  for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
  Retrieval](https://huggingface.co/papers/2010.00904).
* **synced\_gpus** (`bool`, *optional*, defaults to `False`) —
  Whether to continue running the while loop until max\_length (needed to avoid deadlocking with
  `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
* **kwargs** (`dict[str, Any]`, *optional*) —
  Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
  forwarded to the `forward` function of the model.

Returns

[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`. The possible
[ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) types are:

* [GenerateEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
* [GenerateBeamEncoderDecoderOutput](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates sequences of token ids.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model’s default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4TConfig

### class transformers.SeamlessM4TConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/configuration_seamless_m4t.py#L24)

( vocab\_size = 256102 t2u\_vocab\_size = 10082 hidden\_size = 1024 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 use\_cache = True max\_position\_embeddings = 1024 is\_encoder\_decoder = True encoder\_layerdrop = 0.05 decoder\_layerdrop = 0.05 activation\_function = 'relu' dropout = 0.1 attention\_dropout = 0.1 activation\_dropout = 0.0 scale\_embedding = True encoder\_layers = 24 encoder\_ffn\_dim = 8192 encoder\_attention\_heads = 16 decoder\_layers = 24 decoder\_ffn\_dim = 8192 decoder\_attention\_heads = 16 decoder\_start\_token\_id = 3 max\_new\_tokens = 256 pad\_token\_id = 0 bos\_token\_id = 2 eos\_token\_id = 3 speech\_encoder\_layers = 24 speech\_encoder\_attention\_heads = 16 speech\_encoder\_intermediate\_size = 4096 speech\_encoder\_hidden\_act = 'swish' speech\_encoder\_dropout = 0.0 add\_adapter = True speech\_encoder\_layerdrop = 0.1 feature\_projection\_input\_dim = 160 num\_conv\_pos\_embeddings = 128 num\_conv\_pos\_embedding\_groups = 16 adaptor\_kernel\_size = 8 adaptor\_stride = 8 adaptor\_dropout = 0.1 num\_adapter\_layers = 1 position\_embeddings\_type = 'relative' rotary\_embedding\_base = 10000 max\_source\_positions = 4096 conv\_depthwise\_kernel\_size = 31 t2u\_bos\_token\_id = 0 t2u\_pad\_token\_id = 1 t2u\_eos\_token\_id = 2 t2u\_decoder\_start\_token\_id = 2 t2u\_max\_new\_tokens = 1024 t2u\_encoder\_layers = 6 t2u\_encoder\_ffn\_dim = 8192 t2u\_encoder\_attention\_heads = 16 t2u\_decoder\_layers = 6 t2u\_decoder\_ffn\_dim = 8192 t2u\_decoder\_attention\_heads = 16 t2u\_max\_position\_embeddings = 2048 sampling\_rate = 16000 upsample\_initial\_channel = 512 upsample\_rates = [5, 4, 4, 2, 2] upsample\_kernel\_sizes = [11, 8, 8, 4, 4] resblock\_kernel\_sizes = [3, 7, 11] resblock\_dilation\_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]] leaky\_relu\_slope = 0.1 unit\_hifi\_gan\_vocab\_size = 10000 unit\_embed\_dim = 1280 lang\_embed\_dim = 256 spkr\_embed\_dim = 256 vocoder\_num\_langs = 36 vocoder\_num\_spkrs = 200 variance\_predictor\_kernel\_size = 3 var\_pred\_dropout = 0.5 vocoder\_offset = 4 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 256102) —
  Vocabulary size of the SeamlessM4T model. Defines the number of different tokens that can be represented by
  the `inputs_ids` passed when calling [~SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel), [~SeamlessM4TForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToSpeech) or
  [~SeamlessM4TForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToText).
* **t2u\_vocab\_size** (`int`, *optional*, defaults to 10082) —
  Unit vocabulary size of the SeamlessM4T model. Defines the number of different unit tokens that can be
  represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [~SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel),
  [~SeamlessM4TForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToSpeech) or [~SeamlessM4TForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToSpeech).

Parameters shared across sub-models

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the “intermediate” layers in the architecture.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **max\_position\_embeddings** (`int`, *optional*, defaults to 1024) —
  The maximum sequence length that this model text encoder and decoder might ever be used with. Typically set
  this to something large just in case (e.g., 512 or 1024 or 2048).
* **is\_encoder\_decoder** (`bool`, *optional*, defaults to `True`) —
  Whether the model is used as an encoder/decoder or not.
* **encoder\_layerdrop** (`float`, *optional*, defaults to 0.05) —
  The LayerDrop probability for the encoders. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **decoder\_layerdrop** (`float`, *optional*, defaults to 0.05) —
  The LayerDrop probability for the decoders. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **activation\_function** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the decoder and feed-forward layers. If string,
  `"gelu"`, `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
* **dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings, encoder, decoder, and pooler.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all attention layers.
* **activation\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all activation layers in the model.
* **scale\_embedding** (`bool`, *optional*, defaults to `True`) —
  Scale embeddings by diving by sqrt(d\_model).

Text encoder and text decoder specific parameters

* **encoder\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer text encoder.
* **encoder\_ffn\_dim** (`int`, *optional*, defaults to 8192) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer text encoder.
* **encoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer text encoder.
* **decoder\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer text decoder.
* **decoder\_ffn\_dim** (`int`, *optional*, defaults to 8192) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer text decoder.
* **decoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer text decoder.
* **decoder\_start\_token\_id** (`int`, *optional*, defaults to 3) —
  If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token. Only
  applied in the text decoder.
* **max\_new\_tokens** (`int`, *optional*, defaults to 256) —
  The maximum numbers of text tokens to generate, ignoring the number of tokens in the prompt.
* **pad\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the *padding* text token. Only applied to the text-decoder model.
* **bos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the *beginning-of-stream* text token. Only applied to the text-decoder model.
* **eos\_token\_id** (`int`, *optional*, defaults to 3) —
  The id of the *end-of-stream* text token. Only applied to the text-decoder model.

Speech encoder specific parameters

* **speech\_encoder\_layers** (`int`, *optional*, defaults to 24) —
  Number of hidden layers in the Transformer speech encoder.
* **speech\_encoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer speech encoder.
* **speech\_encoder\_intermediate\_size** (`int`, *optional*, defaults to 4096) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer speech encoder.
* **speech\_encoder\_hidden\_act** (`str` or `function`, *optional*, defaults to `"swish"`) —
  The non-linear activation function (function or string) in the speech encoder. If string, `"gelu"`,
  `"relu"`, `"selu"`, `"swish"` and `"gelu_new"` are supported.
* **speech\_encoder\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout probability for all layers in the speech encoder.
* **add\_adapter** (`bool`, *optional*, defaults to `True`) —
  Add an adapter layer on top of the speech encoder.
* **speech\_encoder\_layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability for the speech encoder. See the [LayerDrop paper](see
  <https://huggingface.co/papers/1909.11556>) for more details.
* **feature\_projection\_input\_dim** (`int`, *optional*, defaults to 160) —
  Input dimension of the input feature projection of the speech encoder, i.e the dimension after processing
  input audios with [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor).
* **num\_conv\_pos\_embeddings** (`int`, *optional*, defaults to 128) —
  Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
  embeddings layer of the speech encoder.
* **num\_conv\_pos\_embedding\_groups** (`int`, *optional*, defaults to 16) —
  Number of groups of 1D convolutional positional embeddings layer of the speech encoder.
* **adaptor\_kernel\_size** (`int`, *optional*, defaults to 8) —
  Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adaptor\_stride** (`int`, *optional*, defaults to 8) —
  Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adaptor\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all layers in the speech adapter.
* **num\_adapter\_layers** (`int`, *optional*, defaults to 1) —
  Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is True`.
* **position\_embeddings\_type** (`str`, *optional*, defaults to `"relative"`) —
  Can be specified to `relative` or `rotary` for relative or rotary position embeddings respectively. If left
  `None` no relative position embedding is applied. Only applied to the speech encoder.
* **rotary\_embedding\_base** (`int`, *optional*, defaults to 10000) —
  If `"rotary"` position embeddings are used, defines the size of the embedding base. Only applied to the
  speech encoder.
* **max\_source\_positions** (`int`, *optional*, defaults to 4096) —
  if `"relative"` position embeddings are used, defines the maximum source input positions. Only applied to
  the speech encoder.
* **conv\_depthwise\_kernel\_size** (`int`, *optional*, defaults to 31) —
  Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.

Text-To-Unit (t2u) model specific parameters

* **t2u\_bos\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the *beginning-of-stream* unit token. Only applied to the text-to-unit seq2seq model.
* **t2u\_pad\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the *padding* unit token. Only applied to the text-to-unit seq2seq model.
* **t2u\_eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the *end-of-stream* unit token. Only applied to the text-to-unit seq2seq model.
* **t2u\_decoder\_start\_token\_id** (`int`, *optional*, defaults to 2) —
  If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token. Only
  applied to the text-to-unit seq2seq model.
* **t2u\_max\_new\_tokens** (`int`, *optional*, defaults to 1024) —
  The maximum numbers of unit tokens to generate, ignoring the number of tokens in the prompt. Only applied
  to the text-to-unit seq2seq model.
* **t2u\_encoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer text-to-unit encoder.
* **t2u\_encoder\_ffn\_dim** (`int`, *optional*, defaults to 8192) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer text-to-unit encoder.
* **t2u\_encoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer text-to-unit encoder.
* **t2u\_decoder\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer text-to-unit decoder.
* **t2u\_decoder\_ffn\_dim** (`int`, *optional*, defaults to 8192) —
  Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer text-to-unit decoder.
* **t2u\_decoder\_attention\_heads** (`int`, *optional*, defaults to 16) —
  Number of attention heads for each attention layer in the Transformer text-to-unit decoder.
* **t2u\_max\_position\_embeddings** (`int`, *optional*, defaults to 2048) —
  The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
  this to something large just in case (e.g., 512 or 1024 or 2048).
  > Hifi-Gan Vocoder specific parameters
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the output audio will be generated, expressed in hertz (Hz).
* **upsample\_initial\_channel** (`int`, *optional*, defaults to 512) —
  The number of input channels into the hifi-gan upsampling network. Applies to the vocoder only.
* **upsample\_rates** (`tuple[int]` or `list[int]`, *optional*, defaults to `[5, 4, 4, 2, 2]`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the vocoder upsampling network.
  The length of *upsample\_rates* defines the number of convolutional layers and has to match the length of
  *upsample\_kernel\_sizes*. Applies to the vocoder only.
* **upsample\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[11, 8, 8, 4, 4]`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the vocoder upsampling
  network. The length of *upsample\_kernel\_sizes* defines the number of convolutional layers and has to match
  the length of *upsample\_rates*. Applies to the vocoder only.
* **resblock\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`) —
  A tuple of integers defining the kernel sizes of the vocoder 1D convolutional layers in the multi-receptive
  field fusion (MRF) module. Applies to the vocoder only.
* **resblock\_dilation\_sizes** (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) —
  A nested tuple of integers defining the dilation rates of the vocoder dilated 1D convolutional layers in
  the multi-receptive field fusion (MRF) module. Applies to the vocoder only.
* **leaky\_relu\_slope** (`float`, *optional*, defaults to 0.1) —
  The angle of the negative slope used by the leaky ReLU activation in the vocoder. Applies to the vocoder
  only.
* **unit\_hifi\_gan\_vocab\_size** (`int`, *optional*, defaults to 10000) —
  Vocabulary size of the SeamlessM4T vocoder. Defines the number of different unit tokens that can be
  represented by the `inputs_ids` passed when calling the vocoder of [~SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel),
  [~SeamlessM4TForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForSpeechToSpeech) or [~SeamlessM4TForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TForTextToSpeech).
* **unit\_embed\_dim** (`int`, *optional*, defaults to 1280) —
  The projection dimension of the input ids given to the hifi-gan vocoder. Applies to the vocoder only.
* **lang\_embed\_dim** (`int`, *optional*, defaults to 256) —
  The projection dimension of the target language given to the hifi-gan vocoder. Applies to the vocoder only.
* **spkr\_embed\_dim** (`int`, *optional*, defaults to 256) —
  The projection dimension of the speaker id given to the hifi-gan vocoder. Applies to the vocoder only.
* **vocoder\_num\_langs** (`int`, *optional*, defaults to 36) —
  Number of langs supported by the vocoder. Might be different from `t2u_num_langs`.
* **vocoder\_num\_spkrs** (`int`, *optional*, defaults to 200) —
  Number of speakers supported by the vocoder.
* **variance\_predictor\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the duration predictor. Applies to the vocoder only.
* **var\_pred\_dropout** (`float`, *optional*, defaults to 0.5) —
  The dropout probability of the duration predictor. Applies to the vocoder only.
* **vocoder\_offset** (`int`, *optional*, defaults to 4) —
  Offset the unit token ids by this number to account for symbol tokens. Applies to the vocoder only.

This is the configuration class to store the configuration of a [~SeamlessM4TModel](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TModel). It is used to instantiate an
SeamlessM4T model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SeamlessM4T
[“facebook/hf-seamless-m4t-medium”](https://huggingface.co/%22facebook/hf-seamless-m4t-medium%22) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

>>> # Initializing a SeamlessM4T "facebook/hf-seamless-m4t-medium" style configuration
>>> configuration = SeamlessM4TConfig()

>>> # Initializing a model from the "facebook/hf-seamless-m4t-medium" style configuration
>>> model = SeamlessM4TModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## SeamlessM4TTokenizer

### class transformers.SeamlessM4TTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L45)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' tokenizer\_file = None src\_lang = 'eng' tgt\_lang = 'fra' sp\_model\_kwargs: typing.Optional[dict[str, typing.Any]] = None additional\_special\_tokens = None add\_prefix\_space = True \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **tokenizer\_file** (`str`, *optional*) —
  The path to a tokenizer file to use instead of the vocab file.
* **src\_lang** (`str`, *optional*, defaults to `"eng"`) —
  The language to use as source language for translation.
* **tgt\_lang** (`str`, *optional*, defaults to `"fra"`) —
  The language to use as target language for translation.
* **sp\_model\_kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments to pass to the model initialization.
* **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, *optional*) —
  A tuple or a list of additional special tokens. Can be used to specify the list of languages that will be
  supported by the tokenizer.
* **add\_prefix\_space** (`bool`, *optional*, defaults to `True`) —
  Whether or not to add an initial space to the input. This allows to treat the leading word just as any
  other word.

Construct a SeamlessM4T tokenizer.

Adapted from [RobertaTokenizer](/docs/transformers/v4.56.2/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.56.2/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on
[SentencePiece](https://github.com/google/sentencepiece).

The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language code> <tokens> <eos>` for target language documents.

Examples:


```
>>> from transformers import SeamlessM4TTokenizer

>>> tokenizer = SeamlessM4TTokenizer.from_pretrained(
...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L212)

( text: typing.Union[str, list[str], list[list[str]]] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]]] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True pad\_to\_multiple\_of: typing.Optional[int] = 2 src\_lang: typing.Optional[str] = None tgt\_lang: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **src\_lang** (`str`, *optional*) —
  A string representing the source language. If not specified, the last `src_lang` specified (either
  during initialization or when calling this tokenizer) will be used.
* **tgt\_lang** (`str`, *optional*) —
  A string representing the target language. If not specified, the last `tgt_lang` specified (either
  during initialization or when calling this tokenizer) will be used.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__).

#### build\_inputs\_with\_special\_tokens

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L342)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs to which the special tokens will be added.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

* `input_ids` (for encoder) `X [eos, src_lang_code]`
* `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
separator.

#### get\_special\_tokens\_mask

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L311)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None already\_has\_special\_tokens: bool = False  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.
* **already\_has\_special\_tokens** (`bool`, *optional*, defaults to `False`) —
  Whether or not the token list is already formatted with special tokens for the model.

Returns

`list[int]`

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L370)

( token\_ids\_0: list token\_ids\_1: typing.Optional[list[int]] = None  ) → `list[int]`

Parameters

* **token\_ids\_0** (`list[int]`) —
  List of IDs.
* **token\_ids\_1** (`list[int]`, *optional*) —
  Optional second list of IDs for sequence pairs.

Returns

`list[int]`

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not
make use of token type ids, therefore a list of zeros is returned.

#### save\_vocabulary

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t.py#L497)

( save\_directory: str filename\_prefix: typing.Optional[str] = None  )

## SeamlessM4TTokenizerFast

### class transformers.SeamlessM4TTokenizerFast

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t_fast.py#L42)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' src\_lang = 'eng' tgt\_lang = 'fra' additional\_special\_tokens = None \*\*kwargs  )

Parameters

* **vocab\_file** (`str`, *optional*) —
  Path to the vocabulary file.
* **tokenizer\_file** (`str`, *optional*) —
  The path to a tokenizer file to use instead of the vocab file.
* **bos\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

  When building a sequence using special tokens, this is not the token that is used for the beginning of
  sequence. The token used is the `cls_token`.
* **eos\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The end of sequence token.

  When building a sequence using special tokens, this is not the token that is used for the end of sequence.
  The token used is the `sep_token`.
* **sep\_token** (`str`, *optional*, defaults to `"</s>"`) —
  The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
  sequence classification or for a text and a question for question answering. It is also used as the last
  token of a sequence built with special tokens.
* **cls\_token** (`str`, *optional*, defaults to `"<s>"`) —
  The classifier token which is used when doing sequence classification (classification of the whole sequence
  instead of per-token classification). It is the first token of the sequence when built with special tokens.
* **unk\_token** (`str`, *optional*, defaults to `"<unk>"`) —
  The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
  token instead.
* **pad\_token** (`str`, *optional*, defaults to `"<pad>"`) —
  The token used for padding, for example when batching sequences of different lengths.
* **src\_lang** (`str`, *optional*, defaults to `"eng"`) —
  The language to use as source language for translation.
* **tgt\_lang** (`str`, *optional*, defaults to `"fra"`) —
  The language to use as target language for translation.
* **additional\_special\_tokens** (tuple or list of `str` or `tokenizers.AddedToken`, *optional*) —
  A tuple or a list of additional special tokens.

Construct a “fast” SeamlessM4T tokenizer (backed by HuggingFace’s *tokenizers* library). Based on
[BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should
refer to this superclass for more information regarding those methods.

The tokenization method is `<language code> <tokens> <eos>` for source language documents, and `<eos> <language code> <tokens> <eos>` for target language documents.

Examples:


```
>>> from transformers import SeamlessM4TTokenizerFast

>>> tokenizer = SeamlessM4TTokenizerFast.from_pretrained(
...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/tokenization_seamless_m4t_fast.py#L372)

( text: typing.Union[str, list[str], list[list[str]]] = None text\_pair: typing.Union[str, list[str], list[list[str]], NoneType] = None text\_target: typing.Union[str, list[str], list[list[str]]] = None text\_pair\_target: typing.Union[str, list[str], list[list[str]], NoneType] = None padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True pad\_to\_multiple\_of: typing.Optional[int] = 2 src\_lang: typing.Optional[str] = None tgt\_lang: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **text\_pair\_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) —
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **pad\_to\_multiple\_of** (`int`, *optional*) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
* **src\_lang** (`str`, *optional*) —
  A string representing the source language. If not specified, the last `src_lang` specified (either
  during initialization or when calling this tokenizer) will be used.
* **tgt\_lang** (`str`, *optional*) —
  A string representing the target language. If not specified, the last `tgt_lang` specified (either
  during initialization or when calling this tokenizer) will be used.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to [PreTrainedTokenizerFast.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__).

## SeamlessM4TFeatureExtractor

### class transformers.SeamlessM4TFeatureExtractor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py#L38)

( feature\_size = 80 sampling\_rate = 16000 num\_mel\_bins = 80 padding\_value = 0.0 stride = 2 \*\*kwargs  )

Parameters

* **feature\_size** (`int`, *optional*, defaults to 80) —
  The feature dimension of the extracted features.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
* **num\_mel\_bins** (`int`, *optional*, defaults to 80) —
  Number of Mel-frequency bins.
* **padding\_value** (`float`, *optional*, defaults to 0.0) —
  The value that is used to fill the padding vectors.
* **stride** (`int`, *optional*, defaults to 2) —
  Stride used to reshape audios from shape (batch\_size,num\_frames,num\_mel\_bins) to
  (batch\_size,num\_frames//stride,num\_mel\_bins\*stride).

Constructs a SeamlessM4T feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.56.2/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains most of the main methods. Users
should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py#L144)

( raw\_speech: typing.Union[numpy.ndarray, list[float], list[numpy.ndarray], list[list[float]]] padding: typing.Union[bool, str, transformers.utils.generic.PaddingStrategy] = True pad\_to\_multiple\_of: typing.Optional[int] = 2 max\_length: typing.Optional[int] = None truncation: bool = False return\_tensors: typing.Union[str, transformers.utils.generic.TensorType, NoneType] = None sampling\_rate: typing.Optional[int] = None return\_attention\_mask: typing.Optional[bool] = None do\_normalize\_per\_mel\_bins: typing.Optional[bool] = True \*\*kwargs  )

Parameters

* **raw\_speech** (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`, —
* **`list[list[float]]`,** `list[list[list[float]]]`) —
  The sequence or batch of sequences to be padded. Each sequence can be a numpy array,
  a torch tensor, a list of float values, a list of numpy arrays, a list of torch tensors,
  a list of list of float values or a list of a list of list of float values.
  If `raw_speech` is a one-dimensional `np.ndarray`, `torch.Tensor` or a `list[float]`, `raw_speech` is
  considered a single-channel, single-sample sound. In all other cases, the first dimension of
  `raw_speech`, whether from an `np.ndarray`, a `torch.Tensor` or a `list[...]`,
  corresponds to the number of samples in the batch, and the number of channels
  (i.e. mono or stereo character) is derived from the other dimensions
  (1D -> single-channel waveform batches; 2D-> stereo-channel waveform batches).
* **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) —
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding
  index) among:
  + `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence if provided).
  + `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  + `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
* **pad\_to\_multiple\_of** (`int`, *optional*, defaults to 2) —
  If set will pad the sequence to a multiple of the provided value.

  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
* **max\_length** (`int`, *optional*) —
  Maximum length of the returned list and optionally padding length (see above).
* **truncation** (`bool`) —
  Activates truncation to cut input sequences longer than *max\_length* to *max\_length*.
* **return\_attention\_mask** (`bool`, *optional*) —
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific feature\_extractor’s default.

  [What are attention masks?](../glossary#attention-mask)

  For SeamlessM4T models, `attention_mask` should always be passed for batched inference, to avoid subtle
  bugs.
* **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.56.2/en/internal/file_utils#transformers.TensorType), *optional*) —
  If set, will return tensors instead of list of python integers. Acceptable values are:
  + `'tf'`: Return TensorFlow `tf.constant` objects.
  + `'pt'`: Return PyTorch `torch.Tensor` objects.
  + `'np'`: Return Numpy `np.ndarray` objects.
* **sampling\_rate** (`int`, *optional*) —
  The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
  `sampling_rate` at the forward call to prevent silent errors.
* **do\_normalize\_per\_mel\_bins** (`bool`, *optional*, defaults to `True`) —
  Whether or not to zero-mean unit-variance normalize the input per mel-channel.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to the tokenizer or the feature
  extractor.

Main method to featurize and prepare for the model one or several sequence(s).

## SeamlessM4TProcessor

### class transformers.SeamlessM4TProcessor

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/processing_seamless_m4t.py#L22)

( feature\_extractor tokenizer  )

Parameters

* **feature\_extractor** ([SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor)) —
  The audio processor is a required input.
* **tokenizer** ([SeamlessM4TTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast)) —
  The tokenizer is a required input.

Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
single processor.

[SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) offers all the functionalities of [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) and
[SeamlessM4TTokenizerFast](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast). See the [**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor.__call__) and [decode()](/docs/transformers/v4.56.2/en/main_classes/processors#transformers.ProcessorMixin.decode) for
more information.

#### \_\_call\_\_

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/processing_seamless_m4t.py#L44)

( text = None audios = None src\_lang = None tgt\_lang = None \*\*kwargs  ) → [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

* **text** (`str`, `list[str]`, `list[list[str]]`) —
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
* **audios** (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) —
  The audio or batch of audios to be prepared. Each audio can be NumPy array or PyTorch tensor. In case
  of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
  and T the sample length of the audio.
* **src\_lang** (`str`, *optional*) —
  The language code of the input texts/audios. If not specified, the last `src_lang` specified will be
  used.
* **tgt\_lang** (`str`, *optional*) —
  The code of the target language. If not specified, the last `tgt_lang` specified will be used.
* **kwargs** (*optional*) —
  Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the
  tokenizer.

Returns

[BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

* **input\_ids** — List of token ids to be fed to a model. Returned when `text` is not `None`.
* **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *“attention\_mask”* is in `self.model_input_names` and if `text` is not
  `None`).
* **input\_features** — Audio input features to be fed to a model. Returned when `audios` is not `None`.

Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `text`
and `kwargs` arguments to SeamlessM4TTokenizerFast’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast.__call__) if `text` is not
`None` to encode the text. To prepare the audio(s), this method forwards the `audios` and `kwrags` arguments to
SeamlessM4TFeatureExtractor’s [**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) if `audios` is not `None`. Please refer
to the docstring of the above two methods for more information.

## SeamlessM4TCodeHifiGan

### class transformers.SeamlessM4TCodeHifiGan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2290)

( config  )

Parameters

* **config** ([SeamlessM4TCodeHifiGan](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TCodeHifiGan)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Code HiFi-GAN vocoder as described in this [repository](https://github.com/facebookresearch/speech-resynthesis).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2366)

( input\_ids: LongTensor spkr\_id: Tensor lang\_id: Tensor  )

Parameters

* **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) —
  Indices of input sequence tokens in the vocabulary.

  Indices can be obtained using [SeamlessM4TTextToUnitForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTextToUnitForConditionalGeneration). [What are input
  IDs?](../glossary#input-ids)
* **spkr\_id** (`int`, *optional*) —
  The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
* **tgt\_lang** (`str`, *optional*) —
  The language id to use as target language for translation.

## SeamlessM4THifiGan

### class transformers.SeamlessM4THifiGan

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2213)

( config: SeamlessM4TConfig  )

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2248)

( input\_embeds: FloatTensor  ) → `torch.FloatTensor`

Parameters

* **spectrogram** (`torch.FloatTensor`) —
  Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length, model_in_dim)`, or un-batched and of shape `(sequence_length, model_in_dim)`. Note that `model_in_dim`
  is the sum of `config.unit_embed_dim`, `config.lang_embed_dim` and `config.spkr_embed_dim`.

Returns

`torch.FloatTensor`

Tensor containing the speech waveform. If the input spectrogram is batched, will be of
shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.

Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
waveform.

## SeamlessM4TTextToUnitModel

### class transformers.SeamlessM4TTextToUnitModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L1889)

( config: SeamlessM4TConfig embed\_tokens\_decoder: typing.Optional[torch.nn.modules.sparse.Embedding] = None  )

Parameters

* **config** ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **embed\_tokens\_decoder** (`nn.Embedding`, *optional*) —
  input embedding of the decoder.

Transformer bare text-to-unit encoder-decoder. The encoder is a `SeamlessM4TEncoder` without embeddings and the decoder is a `SeamlessM4TDecoder`.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## SeamlessM4TTextToUnitForConditionalGeneration

### class transformers.SeamlessM4TTextToUnitForConditionalGeneration

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L1982)

( config: SeamlessM4TConfig embed\_tokens\_decoder: typing.Optional[torch.nn.modules.sparse.Embedding] = None  )

Parameters

* **config** ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **embed\_tokens\_decoder** (`nn.Embedding`, *optional*) —
  input embedding of the decoder.

Transformer text-to-unit encoder-decoder with a language model head. The base encoder-decoder model is a `SeamlessM4TTextToUnit`.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2026)

( input\_ids: typing.Optional[torch.LongTensor] = None attention\_mask: typing.Optional[torch.Tensor] = None decoder\_input\_ids: typing.Optional[torch.LongTensor] = None decoder\_attention\_mask: typing.Optional[torch.LongTensor] = None encoder\_outputs: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None past\_key\_values: typing.Optional[tuple[tuple[torch.FloatTensor]]] = None inputs\_embeds: typing.Optional[torch.FloatTensor] = None decoder\_inputs\_embeds: typing.Optional[torch.FloatTensor] = None labels: typing.Optional[torch.LongTensor] = None use\_cache: typing.Optional[bool] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None cache\_position: typing.Optional[torch.Tensor] = None  ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

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
* **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Indices of decoder input sequence tokens in the vocabulary.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.**call**()](/docs/transformers/v4.56.2/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are decoder input IDs?](../glossary#decoder-input-ids)

  Bart uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
  is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

  For translation and summarization training, `decoder_input_ids` should be provided. If no
  `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right
  for denoising pre-training following the paper.
* **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) —
  Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
  be used by default.

  If you want to change padding behavior, you should read `modeling_bart._prepare_decoder_attention_mask`
  and modify to your needs. See diagram 1 in [the paper](https://huggingface.co/papers/1910.13461) for more
  information on the default strategy.
* **encoder\_outputs** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
  `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
  hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
* **past\_key\_values** (`tuple[tuple[torch.FloatTensor]]`, *optional*) —
  Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
  returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  Only [Cache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don’t
  have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids`
  of shape `(batch_size, sequence_length)`.
* **inputs\_embeds** (`torch.FloatTensor` of shape`(batch_size, sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
  is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
  model’s internal embedding lookup matrix.
* **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) —
  Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
  representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
  input (see `past_key_values`). This is useful if you want more control over how to convert
  `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.

  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
  of `inputs_embeds`.
* **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) —
  Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
  loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
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

[transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([SeamlessM4TConfig](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TConfig)) and inputs.

* **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) — Language modeling loss.
* **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
* **past\_key\_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) — It is a [EncoderDecoderCache](/docs/transformers/v4.56.2/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
* **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
* **decoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
* **cross\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
* **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
* **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
* **encoder\_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

The [SeamlessM4TTextToUnitForConditionalGeneration](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TTextToUnitForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/seamless_m4t.md)
