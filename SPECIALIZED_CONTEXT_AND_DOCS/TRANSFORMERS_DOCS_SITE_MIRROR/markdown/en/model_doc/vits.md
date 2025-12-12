# VITS

[VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)](https://huggingface.co/papers/2106.06103) is a end-to-end speech synthesis model, simplifying the traditional two-stage text-to-speech (TTS) systems. It's unique because it directly synthesizes speech from text using variational inference, adversarial learning, and normalizing flows to produce natural and expressive speech with diverse rhythms and intonations.

You can find all the original VITS checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mms-tts) organization.

> [!TIP]
> Click on the VITS models in the right sidebar for more examples of how to apply VITS.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/main/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/main/en/model_doc/auto#transformers.AutoModel) class.

```python
import torch
from transformers import pipeline, set_seed
from scipy.io.wavfile import write

set_seed(555)

pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-eng",
    dtype=torch.float16,
    device=0
)

speech = pipe("Hello, my dog is cute")

# Extract audio data and sampling rate
audio_data = speech["audio"]
sampling_rate = speech["sampling_rate"]

# Save as WAV file
write("hello.wav", sampling_rate, audio_data.squeeze())
```

```python
import torch
import scipy
from IPython.display import Audio
from transformers import AutoTokenizer, VitsModel, set_seed

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng", device_map="auto", dtype=torch.float16)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(model.device)

set_seed(555)

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]
scipy.io.wavfile.write("hello.wav", rate=model.config.sampling_rate, data=waveform)

# display in Colab notebook
Audio(waveform, rate=model.config.sampling_rate)
```

## Notes

- Set a seed for reproducibility because VITS synthesizes speech non-deterministically.
- For languages with non-Roman alphabets (Korean, Arabic, etc.), install the [uroman](https://github.com/isi-nlp/uroman) package to preprocess the text inputs to the Roman alphabet. You can check if the tokenizer requires uroman as shown below.

   ```py
   # pip install -U uroman
   from transformers import VitsTokenizer

   tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
   print(tokenizer.is_uroman)
   ```

   If your language requires uroman, the tokenizer automatically applies it to the text inputs. Python >= 3.10 doesn't require any additional preprocessing steps. For Python >> from transformers import VitsModel, VitsConfig

>>> # Initializing a "facebook/mms-tts-eng" style configuration
>>> configuration = VitsConfig()

>>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
>>> model = VitsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

**Parameters:**

vocab_size (`int`, *optional*, defaults to 38) : Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by the `inputs_ids` passed to the forward method of [VitsModel](/docs/transformers/main/en/model_doc/vits#transformers.VitsModel).

hidden_size (`int`, *optional*, defaults to 192) : Dimensionality of the text encoder layers.

num_hidden_layers (`int`, *optional*, defaults to 6) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to 2) : Number of attention heads for each attention layer in the Transformer encoder.

window_size (`int`, *optional*, defaults to 4) : Window size for the relative positional embeddings in the attention layers of the Transformer encoder.

use_bias (`bool`, *optional*, defaults to `True`) : Whether to use bias in the key, query, value projection layers in the Transformer encoder.

ffn_dim (`int`, *optional*, defaults to 768) : Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.

layerdrop (`float`, *optional*, defaults to 0.1) : The LayerDrop probability for the encoder. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

ffn_kernel_size (`int`, *optional*, defaults to 3) : Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.

flow_size (`int`, *optional*, defaults to 192) : Dimensionality of the flow layers.

spectrogram_bins (`int`, *optional*, defaults to 513) : Number of frequency bins in the target spectrogram.

hidden_act (`str` or `function`, *optional*, defaults to `"relu"`) : The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.

hidden_dropout (`float`, *optional*, defaults to 0.1) : The dropout probability for all fully connected layers in the embeddings and encoder.

attention_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for the attention probabilities.

activation_dropout (`float`, *optional*, defaults to 0.1) : The dropout ratio for activations inside the fully connected layer.

initializer_range (`float`, *optional*, defaults to 0.02) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to 1e-05) : The epsilon used by the layer normalization layers.

use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`) : Whether to use the stochastic duration prediction module or the regular duration predictor.

num_speakers (`int`, *optional*, defaults to 1) : Number of speakers if this is a multi-speaker model.

speaker_embedding_size (`int`, *optional*, defaults to 0) : Number of channels used by the speaker embeddings. Is zero for single-speaker models.

upsample_initial_channel (`int`, *optional*, defaults to 512) : The number of input channels into the HiFi-GAN upsampling network.

upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`) : A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_rates` defines the number of convolutional layers and has to match the length of `upsample_kernel_sizes`.

upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`) : A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match the length of `upsample_rates`.

resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`) : A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.

resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) : A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.

leaky_relu_slope (`float`, *optional*, defaults to 0.1) : The angle of the negative slope used by the leaky ReLU activation.

depth_separable_channels (`int`, *optional*, defaults to 2) : Number of channels to use in each depth-separable block.

depth_separable_num_layers (`int`, *optional*, defaults to 3) : Number of convolutional layers to use in each depth-separable block.

duration_predictor_flow_bins (`int`, *optional*, defaults to 10) : Number of channels to map using the unonstrained rational spline in the duration predictor model.

duration_predictor_tail_bound (`float`, *optional*, defaults to 5.0) : Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor model.

duration_predictor_kernel_size (`int`, *optional*, defaults to 3) : Kernel size of the 1D convolution layers used in the duration predictor model.

duration_predictor_dropout (`float`, *optional*, defaults to 0.5) : The dropout ratio for the duration predictor model.

duration_predictor_num_flows (`int`, *optional*, defaults to 4) : Number of flow stages used by the duration predictor model.

duration_predictor_filter_channels (`int`, *optional*, defaults to 256) : Number of channels for the convolution layers used in the duration predictor model.

prior_encoder_num_flows (`int`, *optional*, defaults to 4) : Number of flow stages used by the prior encoder flow model.

prior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 4) : Number of WaveNet layers used by the prior encoder flow model.

posterior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 16) : Number of WaveNet layers used by the posterior encoder model.

wavenet_kernel_size (`int`, *optional*, defaults to 5) : Kernel size of the 1D convolution layers used in the WaveNet model.

wavenet_dilation_rate (`int`, *optional*, defaults to 1) : Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.

wavenet_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the WaveNet layers.

speaking_rate (`float`, *optional*, defaults to 1.0) : Speaking rate. Larger values give faster synthesised speech.

noise_scale (`float`, *optional*, defaults to 0.667) : How random the speech prediction is. Larger values create more variation in the predicted speech.

noise_scale_duration (`float`, *optional*, defaults to 0.8) : How random the duration prediction is. Larger values create more variation in the predicted durations.

sampling_rate (`int`, *optional*, defaults to 16000) : The sampling rate at which the output audio waveform is digitalized expressed in hertz (Hz).

## VitsTokenizer[[transformers.VitsTokenizer]]

#### transformers.VitsTokenizer[[transformers.VitsTokenizer]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vits/tokenization_vits.py#L47)

Construct a VITS tokenizer. Also supports MMS-TTS.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/main/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

__call__transformers.VitsTokenizer.__call__https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L2469[{"name": "text", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "text_target", "val": ": Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput], None] = None"}, {"name": "text_pair_target", "val": ": Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None"}, {"name": "add_special_tokens", "val": ": bool = True"}, {"name": "padding", "val": ": Union[bool, str, PaddingStrategy] = False"}, {"name": "truncation", "val": ": Union[bool, str, TruncationStrategy, None] = None"}, {"name": "max_length", "val": ": Optional[int] = None"}, {"name": "stride", "val": ": int = 0"}, {"name": "is_split_into_words", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": Optional[int] = None"}, {"name": "padding_side", "val": ": Optional[str] = None"}, {"name": "return_tensors", "val": ": Optional[Union[str, TensorType]] = None"}, {"name": "return_token_type_ids", "val": ": Optional[bool] = None"}, {"name": "return_attention_mask", "val": ": Optional[bool] = None"}, {"name": "return_overflowing_tokens", "val": ": bool = False"}, {"name": "return_special_tokens_mask", "val": ": bool = False"}, {"name": "return_offsets_mapping", "val": ": bool = False"}, {"name": "return_length", "val": ": bool = False"}, {"name": "verbose", "val": ": bool = True"}, {"name": "tokenizer_kwargs", "val": ": Optional[dict[str, Any]] = None"}, {"name": "**kwargs", "val": ""}]- **text** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
  (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
  `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **text_pair_target** (`str`, `list[str]`, `list[list[str]]`, *optional*) --
  The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a
  list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized),
  you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
- **tokenizer_kwargs** (`dict[str, Any]`, *optional*) --
  Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and
  other kwargs, with explicit parameters taking precedence.

- **add_special_tokens** (`bool`, *optional*, defaults to `True`) --
  Whether or not to add special tokens when encoding the sequences. This will use the underlying
  `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
  automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens
  automatically.
- **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/main/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) --
  Activates and controls padding. Accepts the following values:

  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
    sequence is provided).
  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
    acceptable input length for the model if that argument is not provided.
  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
    lengths).
- **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/main/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) --
  Activates and controls truncation. Accepts the following values:

  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
    to the maximum acceptable input length for the model if that argument is not provided. This will
    truncate token by token, removing a token from the longest sequence in the pair if a pair of
    sequences (or a batch of pairs) is provided.
  - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
    maximum acceptable input length for the model if that argument is not provided. This will only
    truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
  - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
    greater than the model maximum admissible input size).
- **max_length** (`int`, *optional*) --
  Controls the maximum length to use by one of the truncation/padding parameters.

  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
  is required by one of the truncation/padding parameters. If the model has no specific maximum input
  length (like XLNet) truncation/padding to a maximum length will be deactivated.
- **stride** (`int`, *optional*, defaults to 0) --
  If set to a number along with `max_length`, the overflowing tokens returned when
  `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
  returned to provide some overlap between truncated and overflowing sequences. The value of this
  argument defines the number of overlapping tokens.
- **is_split_into_words** (`bool`, *optional*, defaults to `False`) --
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize. This is useful for NER or token classification.
- **pad_to_multiple_of** (`int`, *optional*) --
  If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
  `>= 7.5` (Volta).
- **padding_side** (`str`, *optional*) --
  The side on which the model should have padding applied. Should be selected between ['right', 'left'].
  Default value is picked from the class attribute of the same name.
- **return_tensors** (`str` or [TensorType](/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) --
  If set, will return tensors instead of list of python integers. Acceptable values are:

  - `'pt'`: Return PyTorch `torch.Tensor` objects.
  - `'np'`: Return Numpy `np.ndarray` objects.

- **return_token_type_ids** (`bool`, *optional*) --
  Whether to return token type IDs. If left to the default, will return the token type IDs according to
  the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are token type IDs?](../glossary#token-type-ids)
- **return_attention_mask** (`bool`, *optional*) --
  Whether to return the attention mask. If left to the default, will return the attention mask according
  to the specific tokenizer's default, defined by the `return_outputs` attribute.

  [What are attention masks?](../glossary#attention-mask)
- **return_overflowing_tokens** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch
  of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead
  of returning overflowing tokens.
- **return_special_tokens_mask** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return special tokens mask information.
- **return_offsets_mapping** (`bool`, *optional*, defaults to `False`) --
  Whether or not to return `(char_start, char_end)` for each token.

  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/main/en/main_classes/tokenizer#transformers.TokenizersBackend), if using
  Python's tokenizer, this method will raise `NotImplementedError`.
- **return_length**  (`bool`, *optional*, defaults to `False`) --
  Whether or not to return the lengths of the encoded inputs.
- **verbose** (`bool`, *optional*, defaults to `True`) --
  Whether or not to print more information and warnings.
- ****kwargs** -- passed to the `self.tokenize()` method0[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

language (`str`, *optional*) : Language identifier.

add_blank (`bool`, *optional*, defaults to `True`) : Whether to insert token id 0 in between the other tokens.

normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the input text by removing all casing and punctuation.

phonemize (`bool`, *optional*, defaults to `True`) : Whether to convert the input text into phonemes.

is_uroman (`bool`, *optional*, defaults to `False`) : Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.

**Returns:**

`[BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding)`

A [BatchEncoding](/docs/transformers/main/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)
#### save_vocabulary[[transformers.VitsTokenizer.save_vocabulary]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vits/tokenization_vits.py#L234)

## VitsModel[[transformers.VitsModel]]

#### transformers.VitsModel[[transformers.VitsModel]]

[Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vits/modeling_vits.py#L1241)

The complete VITS model, for text-to-speech synthesis.

This model inherits from [PreTrainedModel](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

forwardtransformers.VitsModel.forwardhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/vits/modeling_vits.py#L1268[{"name": "input_ids", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "attention_mask", "val": ": typing.Optional[torch.Tensor] = None"}, {"name": "speaker_id", "val": ": typing.Optional[int] = None"}, {"name": "output_attentions", "val": ": typing.Optional[bool] = None"}, {"name": "output_hidden_states", "val": ": typing.Optional[bool] = None"}, {"name": "return_dict", "val": ": typing.Optional[bool] = None"}, {"name": "labels", "val": ": typing.Optional[torch.FloatTensor] = None"}, {"name": "**kwargs", "val": ""}]- **input_ids** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

  Indices can be obtained using [AutoTokenizer](/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and
  [PreTrainedTokenizer.__call__()](/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.

  [What are input IDs?](../glossary#input-ids)
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) --
  Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

  [What are attention masks?](../glossary#attention-mask)
- **speaker_id** (`int`, *optional*) --
  Which speaker embedding to use. Only used for multispeaker models.
- **output_attentions** (`bool`, *optional*) --
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
- **output_hidden_states** (`bool`, *optional*) --
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
- **return_dict** (`bool`, *optional*) --
  Whether or not to return a [ModelOutput](/docs/transformers/main/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
- **labels** (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*) --
  Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
  computation.0`transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)`A `transformers.models.vits.modeling_vits.VitsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VitsConfig](/docs/transformers/main/en/model_doc/vits#transformers.VitsConfig)) and inputs.

- **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- The final audio waveform predicted by the model.
- **sequence_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) -- The length in samples of each element in the `waveform` batch.
- **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) -- The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
  GAN decoder model to obtain the final audio waveform.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
The [VitsModel](/docs/transformers/main/en/model_doc/vits#transformers.VitsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:

```python
>>> from transformers import VitsTokenizer, VitsModel, set_seed
>>> import torch

>>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
>>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

>>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

>>> set_seed(555)  # make deterministic

>>> with torch.no_grad():
...     outputs = model(inputs["input_ids"])
>>> outputs.waveform.shape
torch.Size([1, 45824])
```

**Parameters:**

config ([VitsConfig](/docs/transformers/main/en/model_doc/vits#transformers.VitsConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

**Returns:**

``transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)``

A `transformers.models.vits.modeling_vits.VitsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VitsConfig](/docs/transformers/main/en/model_doc/vits#transformers.VitsConfig)) and inputs.

- **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- The final audio waveform predicted by the model.
- **sequence_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) -- The length in samples of each element in the `waveform` batch.
- **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) -- The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
  GAN decoder model to obtain the final audio waveform.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
