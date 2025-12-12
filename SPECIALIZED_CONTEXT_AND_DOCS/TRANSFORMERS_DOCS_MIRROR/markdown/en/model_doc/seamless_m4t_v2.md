*This model was released on 2023-12-08 and added to Hugging Face Transformers on 2023-11-30.*

# SeamlessM4T-v2

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

## Overview

The SeamlessM4T-v2 model was proposed in [Seamless: Multilingual Expressive and Streaming Speech Translation](https://huggingface.co/papers/2312.05187) by the Seamless Communication team from Meta AI.

SeamlessM4T-v2 is a collection of models designed to provide high quality translation, allowing people from different linguistic communities to communicate effortlessly through speech and text. It is an improvement on the [previous version](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t). For more details on the differences between v1 and v2, refer to section [Difference with SeamlessM4T-v1](#difference-with-seamlessm4t-v1).

SeamlessM4T-v2 enables multiple tasks without relying on separate models:

* Speech-to-speech translation (S2ST)
* Speech-to-text translation (S2TT)
* Text-to-speech translation (T2ST)
* Text-to-text translation (T2TT)
* Automatic speech recognition (ASR)

[SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) can perform all the above tasks, but each task also has its own dedicated sub-model.

The abstract from the paper is the following:

*Recent advancements in automatic speech translation have dramatically expanded language coverage, improved multimodal capabilities, and enabled a wide range of tasks and functionalities. That said, large-scale automatic speech translation systems today lack key features that help machine-mediated communication feel seamless when compared to human-to-human dialogue. In this work, we introduce a family of models that enable end-to-end expressive and multilingual translations in a streaming fashion. First, we contribute an improved version of the massively multilingual and multimodal SeamlessM4T model—SeamlessM4T v2. This newer model, incorporating an updated UnitY2 framework, was trained on more low-resource language data. The expanded version of SeamlessAlign adds 114,800 hours of automatically aligned data for a total of 76 languages. SeamlessM4T v2 provides the foundation on which our two newest models, SeamlessExpressive and SeamlessStreaming, are initiated. SeamlessExpressive enables translation that preserves vocal styles and prosody. Compared to previous efforts in expressive speech research, our work addresses certain underexplored aspects of prosody, such as speech rate and pauses, while also preserving the style of one’s voice. As for SeamlessStreaming, our model leverages the Efficient Monotonic Multihead Attention (EMMA) mechanism to generate low-latency target translations without waiting for complete source utterances. As the first of its kind, SeamlessStreaming enables simultaneous speech-to-speech/text translation for multiple source and target languages. To understand the performance of these models, we combined novel and modified versions of existing automatic metrics to evaluate prosody, latency, and robustness. For human evaluations, we adapted existing protocols tailored for measuring the most relevant attributes in the preservation of meaning, naturalness, and expressivity. To ensure that our models can be used safely and responsibly, we implemented the first known red-teaming effort for multimodal machine translation, a system for the detection and mitigation of added toxicity, a systematic evaluation of gender bias, and an inaudible localized watermarking mechanism designed to dampen the impact of deepfakes. Consequently, we bring major components from SeamlessExpressive and SeamlessStreaming together to form Seamless, the first publicly available system that unlocks expressive cross-lingual communication in real-time. In sum, Seamless gives us a pivotal look at the technical foundation needed to turn the Universal Speech Translator from a science fiction concept into a real-world technology. Finally, contributions in this work—including models, code, and a watermark detector—are publicly released and accessible at the link below.*

## Usage

In the following example, we’ll load an Arabic audio sample and an English text sample and convert them into Russian speech and French text.

First, load the processor and a checkpoint of the model:


```
>>> from transformers import AutoProcessor, SeamlessM4Tv2Model

>>> processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
>>> model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
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

>>> # now, process some English text as well
>>> text_inputs = processor(text = "Hello, my dog is cute", src_lang="eng", return_tensors="pt")
```

### Speech

[SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) can *seamlessly* generate text or speech with few or no changes. Let’s target Russian voice translation:


```
>>> audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
>>> audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()
```

With basically the same code, I’ve translated English text and Arabic speech to Russian speech samples.

### Text

Similarly, you can generate translated text from audio files or from text with the same model. You only have to pass `generate_speech=False` to [SeamlessM4Tv2Model.generate()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model.generate).
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

[SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint.
For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code:


```
>>> from transformers import SeamlessM4Tv2ForSpeechToSpeech
>>> model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
```

Or you can replace the text-to-text generation snippet with the model dedicated to the T2TT task, you only have to remove `generate_speech=False`.


```
>>> from transformers import SeamlessM4Tv2ForTextToText
>>> model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
```

Feel free to try out [SeamlessM4Tv2ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToText) and [SeamlessM4Tv2ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToSpeech) as well.

#### 2. Change the speaker identity

You have the possibility to change the speaker used for speech synthesis with the `speaker_id` argument. Some `speaker_id` works better than other for some languages!

#### 3. Change the generation strategy

You can use different [generation strategies](../generation_strategies) for text generation, e.g `.generate(input_ids=input_ids, text_num_beams=4, text_do_sample=True)` which will perform multinomial beam-search decoding on the text model. Note that speech generation only supports greedy - by default - or multinomial sampling, which can be used with e.g. `.generate(..., speech_do_sample=True, speech_temperature=0.6)`.

#### 4. Generate speech and text at the same time

Use `return_intermediate_token_ids=True` with [SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model) to return both speech and text !

## Model architecture

SeamlessM4T-v2 features a versatile architecture that smoothly handles the sequential generation of text and speech. This setup comprises two sequence-to-sequence (seq2seq) models. The first model translates the input modality into translated text, while the second model generates speech tokens, known as “unit tokens,” from the translated text.

Each modality has its own dedicated encoder with a unique architecture. Additionally, for speech output, a vocoder inspired by the [HiFi-GAN](https://huggingface.co/papers/2010.05646) architecture is placed on top of the second seq2seq model.

### Difference with SeamlessM4T-v1

The architecture of this new version differs from the first in a few aspects:

#### Improvements on the second-pass model

The second seq2seq model, named text-to-unit model, is now non-auto regressive, meaning that it computes units in a **single forward pass**. This achievement is made possible by:

* the use of **character-level embeddings**, meaning that each character of the predicted translated text has its own embeddings, which are then used to predict the unit tokens.
* the use of an intermediate duration predictor, that predicts speech duration at the **character-level** on the predicted translated text.
* the use of a new text-to-unit decoder mixing convolutions and self-attention to handle longer context.

#### Difference in the speech encoder

The speech encoder, which is used during the first-pass generation process to predict the translated text, differs mainly from the previous speech encoder through these mechanisms:

* the use of chunked attention mask to prevent attention across chunks, ensuring that each position attends only to positions within its own chunk and a fixed number of previous chunks.
* the use of relative position embeddings which only considers distance between sequence elements rather than absolute positions. Please refer to [Self-Attentionwith Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155) for more details.
* the use of a causal depth-wise convolution instead of a non-causal one.

### Generation process

Here’s how the generation process works:

* Input text or speech is processed through its specific encoder.
* A decoder creates text tokens in the desired language.
* If speech generation is required, the second seq2seq model, generates unit tokens in an non auto-regressive way.
* These unit tokens are then passed through the final vocoder to produce the actual speech.

This model was contributed by [ylacombe](https://huggingface.co/ylacombe). The original code can be found [here](https://github.com/facebookresearch/seamless_communication).

## SeamlessM4Tv2Model

### class transformers.SeamlessM4Tv2Model

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3922)

( config current\_modality = 'text'  )

Parameters

* **config** ([SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
* **current\_modality** (`str`, *optional*, defaults to `"text"`) —
  Default modality. Used to initialize the model.

The original SeamlessM4Tv2 Model transformer which can be used for every tasks available (S2ST, S2TT, T2TT, T2ST).

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L4139)

( input\_ids: typing.Optional[torch.Tensor] = None input\_features: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None speaker\_id: typing.Optional[int] = 0 generate\_speech: typing.Optional[bool] = True \*\*kwargs  ) → `Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor], ModelOutput]`

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
* **speaker\_id** (`int`, *optional*, defaults to 0) —
  The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
* **generate\_speech** (`bool`, *optional*, defaults to `True`) —
  If `False`, will only returns the text tokens and won’t generate speech.
* **kwargs** (*optional*) —
  Remaining dictioy of keyword arguments that will be passed to [GenerationMixin.generate()](/docs/transformers/v4.56.2/en/main_classes/text_generation#transformers.GenerationMixin.generate). Keyword
  arguments are of two types:
  + Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
    except for `decoder_input_ids` which will only be passed through the text components.
  + With a *text\_* or *speech\_* prefix, they will be input for the `generate` method of the
    text model and speech model respectively. It has the priority over the keywords without a prefix.

  This means you can, for example, specify a generation strategy for one generation but not for the
  other.

Returns

`Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor], ModelOutput]`

* If `generate_speech` and `return_intermediate_token_ids`, returns `SeamlessM4Tv2GenerationOutput`.
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

## SeamlessM4Tv2ForTextToSpeech

### class transformers.SeamlessM4Tv2ForTextToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3193)

( config: SeamlessM4Tv2Config  )

Parameters

* **config** ([SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text-to-speech SeamlessM4Tv2 Model transformer which can be used for T2ST.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3350)

( input\_ids: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None speaker\_id: typing.Optional[int] = 0 \*\*kwargs  ) → `Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor]]`

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
* **speaker\_id** (`int`, *optional*, defaults to 0) —
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

`Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor]]`

* If `return_intermediate_token_ids`, returns `SeamlessM4Tv2GenerationOutput`.
* If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.

Generates translated audio waveforms.

This method successively calls the `.generate` function of two different sub-models. You can specify keyword
arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
that will be passed to one of them.

For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4Tv2ForSpeechToSpeech

### class transformers.SeamlessM4Tv2ForSpeechToSpeech

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3554)

( config  )

Parameters

* **config** ([SeamlessM4Tv2ForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToSpeech)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The speech-to-speech SeamlessM4Tv2 Model transformer which can be used for S2ST.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3712)

( input\_features: typing.Optional[torch.Tensor] = None return\_intermediate\_token\_ids: typing.Optional[bool] = None tgt\_lang: typing.Optional[str] = None speaker\_id: typing.Optional[int] = 0 \*\*kwargs  ) → `Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor]]`

Parameters

* **input\_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_banks)`) —
  Input audio features. This should be returned by the [SeamlessM4TFeatureExtractor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor) class or the
  [SeamlessM4TProcessor](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TProcessor) class. See [SeamlessM4TFeatureExtractor.**call**()](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor.__call__) for details.
* **return\_intermediate\_token\_ids** (`bool`, *optional*) —
  If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
  to get translated text alongside the audio.
* **tgt\_lang** (`str`, *optional*) —
  The language to use as target language for translation.
* **speaker\_id** (`int`, *optional*, defaults to 0) —
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

`Union[SeamlessM4Tv2GenerationOutput, tuple[Tensor]]`

* If `return_intermediate_token_ids`, returns `SeamlessM4Tv2GenerationOutput`.
* If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size, sequence_length)` and `waveform_lengths` which gives the length of each sample.

Generates translated audio waveforms.

This method successively calls the `.generate` function of two different sub-models. You can specify keyword
arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
that will be passed to one of them.

For example, calling `.generate(input_features, num_beams=4, speech_do_sample=True)` will successively perform
beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

For an overview of generation strategies and code examples, check out the [following
guide](./generation_strategies).

## SeamlessM4Tv2ForTextToText

### class transformers.SeamlessM4Tv2ForTextToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2667)

( config: SeamlessM4Tv2Config  )

Parameters

* **config** ([SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text-to-text SeamlessM4Tv2 Model transformer which can be used for T2TT.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2709)

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
elements depending on the configuration ([SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config)) and inputs.

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

The [SeamlessM4Tv2ForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


#### generate

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2807)

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

## SeamlessM4Tv2ForSpeechToText

### class transformers.SeamlessM4Tv2ForSpeechToText

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2924)

( config: SeamlessM4Tv2Config  )

Parameters

* **config** ([SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The speech-to-text SeamlessM4Tv2 Model transformer which can be used for S2TT.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2967)

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
elements depending on the configuration ([SeamlessM4Tv2Config](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Config)) and inputs.

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

The [SeamlessM4Tv2ForSpeechToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
>>> from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("""")
>>> model = SeamlessM4Tv2ForSpeechToText.from_pretrained("""")

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

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3074)

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

## SeamlessM4Tv2Config

### class transformers.SeamlessM4Tv2Config

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/seamless_m4t_v2/configuration_seamless_m4t_v2.py#L24)

( vocab\_size = 256102 t2u\_vocab\_size = 10082 char\_vocab\_size = 10943 hidden\_size = 1024 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 use\_cache = True max\_position\_embeddings = 4096 is\_encoder\_decoder = True encoder\_layerdrop = 0.05 decoder\_layerdrop = 0.05 activation\_function = 'relu' dropout = 0.1 attention\_dropout = 0.1 activation\_dropout = 0.0 scale\_embedding = True encoder\_layers = 24 encoder\_ffn\_dim = 8192 encoder\_attention\_heads = 16 decoder\_layers = 24 decoder\_ffn\_dim = 8192 decoder\_attention\_heads = 16 decoder\_start\_token\_id = 3 max\_new\_tokens = 256 pad\_token\_id = 0 bos\_token\_id = 2 eos\_token\_id = 3 speech\_encoder\_layers = 24 speech\_encoder\_attention\_heads = 16 speech\_encoder\_intermediate\_size = 4096 speech\_encoder\_hidden\_act = 'swish' speech\_encoder\_dropout = 0.0 add\_adapter = True speech\_encoder\_layerdrop = 0.1 feature\_projection\_input\_dim = 160 adaptor\_kernel\_size = 8 adaptor\_stride = 8 adaptor\_dropout = 0.1 num\_adapter\_layers = 1 position\_embeddings\_type = 'relative\_key' conv\_depthwise\_kernel\_size = 31 left\_max\_position\_embeddings = 64 right\_max\_position\_embeddings = 8 speech\_encoder\_chunk\_size = 20000 speech\_encoder\_left\_chunk\_num = 128 t2u\_bos\_token\_id = 0 t2u\_pad\_token\_id = 1 t2u\_eos\_token\_id = 2 t2u\_encoder\_layers = 6 t2u\_encoder\_ffn\_dim = 8192 t2u\_encoder\_attention\_heads = 16 t2u\_decoder\_layers = 6 t2u\_decoder\_ffn\_dim = 8192 t2u\_decoder\_attention\_heads = 16 t2u\_max\_position\_embeddings = 4096 t2u\_variance\_predictor\_embed\_dim = 1024 t2u\_variance\_predictor\_hidden\_dim = 256 t2u\_variance\_predictor\_kernel\_size = 3 t2u\_variance\_pred\_dropout = 0.5 sampling\_rate = 16000 upsample\_initial\_channel = 512 upsample\_rates = [5, 4, 4, 2, 2] upsample\_kernel\_sizes = [11, 8, 8, 4, 4] resblock\_kernel\_sizes = [3, 7, 11] resblock\_dilation\_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]] leaky\_relu\_slope = 0.1 unit\_hifi\_gan\_vocab\_size = 10000 unit\_embed\_dim = 1280 lang\_embed\_dim = 256 spkr\_embed\_dim = 256 vocoder\_num\_langs = 36 vocoder\_num\_spkrs = 200 variance\_predictor\_kernel\_size = 3 var\_pred\_dropout = 0.5 vocoder\_offset = 4 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 256102) —
  Vocabulary size of the text modality of the SeamlessM4Tv2 model. Defines the number of different tokens
  that can be represented by the `inputs_ids` passed when calling [~SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model),
  [~SeamlessM4Tv2ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToSpeech) or [~SeamlessM4Tv2ForTextToText](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToText).
* **t2u\_vocab\_size** (`int`, *optional*, defaults to 10082) —
  Unit vocabulary size of the SeamlessM4Tv2 model. Defines the number of different “unit tokens” that can be
  represented by the `inputs_ids` passed when calling the Text-To-Units sub-model of [~SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model),
  [~SeamlessM4Tv2ForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToSpeech) or [~SeamlessM4Tv2ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToSpeech).
* **char\_vocab\_size** (`int`, *optional*, defaults to 10943) —
  Character vocabulary size of the SeamlessM4Tv2 model. Defines the number of different character tokens that
  can be represented by the `char_inputs_ids` passed when calling the Text-To-Units sub-model of
  [~SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model), [~SeamlessM4Tv2ForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToSpeech) or [~SeamlessM4Tv2ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToSpeech).

Parameters shared across sub-models

* **hidden\_size** (`int`, *optional*, defaults to 1024) —
  Dimensionality of the “intermediate” layers in the architecture.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **use\_cache** (`bool`, *optional*, defaults to `True`) —
  Whether or not the model should return the last key/values attentions (not used by all models).
* **max\_position\_embeddings** (`int`, *optional*, defaults to 4096) —
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
* **adaptor\_kernel\_size** (`int`, *optional*, defaults to 8) —
  Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adaptor\_stride** (`int`, *optional*, defaults to 8) —
  Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
* **adaptor\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all layers in the speech adapter.
* **num\_adapter\_layers** (`int`, *optional*, defaults to 1) —
  Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is True`.
* **position\_embeddings\_type** (`str`, *optional*, defaults to `"relative_key"`) —
  Can be specified to `relative_key`. If left to `None`, no relative position embedding is applied. Only
  applied to the speech encoder. For more information on `"relative_key"`, please refer to [Self-Attention
  with Relative Position Representations (Shaw et al.)](https://huggingface.co/papers/1803.02155).
* **conv\_depthwise\_kernel\_size** (`int`, *optional*, defaults to 31) —
  Kernel size of convolutional depthwise 1D layer in Conformer blocks. Only applied to the speech encoder.
* **left\_max\_position\_embeddings** (`int`, *optional*, defaults to 64) —
  The left clipping value for relative positions.
* **right\_max\_position\_embeddings** (`int`, *optional*, defaults to 8) —
  The right clipping value for relative positions.
* **speech\_encoder\_chunk\_size** (`int`, *optional*, defaults to 20000) — The size of each attention chunk.
* **speech\_encoder\_left\_chunk\_num** (`int`, *optional*, defaults to 128) —
  Number of chunks on the left up to which lookahead is allowed.

Text-To-Unit (t2u) model specific parameters

* **t2u\_bos\_token\_id** (`int`, *optional*, defaults to 0) —
  The id of the *beginning-of-stream* unit token. Only applied to the text-to-unit seq2seq model.
* **t2u\_pad\_token\_id** (`int`, *optional*, defaults to 1) —
  The id of the *padding* unit token. Only applied to the text-to-unit seq2seq model.
* **t2u\_eos\_token\_id** (`int`, *optional*, defaults to 2) —
  The id of the *end-of-stream* unit token. Only applied to the text-to-unit seq2seq model.
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
* **t2u\_max\_position\_embeddings** (`int`, *optional*, defaults to 4096) —
  The maximum sequence length that this model text-to-unit component might ever be used with. Typically set
  this to something large just in case (e.g., 512 or 1024 or 2048).
* **t2u\_variance\_predictor\_embed\_dim** (`int`, *optional*, defaults to 1024) —
  The projection dimension of the text-to-unit’s duration predictor.
* **t2u\_variance\_predictor\_hidden\_dim** (`int`, *optional*, defaults to 256) —
  Internal dimension of the text-to-unit’s duration predictor.
* **t2u\_variance\_predictor\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the convolutional layers of the text-to-unit’s duration predictor.
* **t2u\_variance\_pred\_dropout** (`float`, *optional*, defaults to 0.5) —
  The dropout probability of the text-to-unit’s duration predictor.
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
  Vocabulary size of the SeamlessM4Tv2 vocoder. Defines the number of different unit tokens that can be
  represented by the `inputs_ids` passed when calling the vocoder of [~SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model),
  [~SeamlessM4Tv2ForSpeechToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForSpeechToSpeech) or [~SeamlessM4Tv2ForTextToSpeech](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2ForTextToSpeech).
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

This is the configuration class to store the configuration of a [~SeamlessM4Tv2Model](/docs/transformers/v4.56.2/en/model_doc/seamless_m4t_v2#transformers.SeamlessM4Tv2Model). It is used to instantiate
an SeamlessM4Tv2 model according to the specified arguments, defining the model architecture. Instantiating a
configuration with the defaults will yield a similar configuration to that of the SeamlessM4Tv2
[""](https://huggingface.co/%22%22) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.


```
>>> from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2Config

>>> # Initializing a SeamlessM4Tv2 "" style configuration
>>> configuration = SeamlessM4Tv2Config()

>>> # Initializing a model from the "" style configuration
>>> model = SeamlessM4Tv2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/seamless_m4t_v2.md)
