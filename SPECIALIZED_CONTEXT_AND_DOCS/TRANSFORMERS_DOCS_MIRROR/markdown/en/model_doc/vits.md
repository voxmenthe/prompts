*This model was released on 2021-06-11 and added to Hugging Face Transformers on 2023-09-01.*

![PyTorch](https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white)

# VITS

[VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)](https://huggingface.co/papers/2106.06103) is a end-to-end speech synthesis model, simplifying the traditional two-stage text-to-speech (TTS) systems. It’s unique because it directly synthesizes speech from text using variational inference, adversarial learning, and normalizing flows to produce natural and expressive speech with diverse rhythms and intonations.

You can find all the original VITS checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mms-tts) organization.

Click on the VITS models in the right sidebar for more examples of how to apply VITS.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v4.56.2/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v4.56.2/en/model_doc/auto#transformers.AutoModel) class.

Pipeline

AutoModel


```
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

## Notes

* Set a seed for reproducibility because VITS synthesizes speech non-deterministically.
* For languages with non-Roman alphabets (Korean, Arabic, etc.), install the [uroman](https://github.com/isi-nlp/uroman) package to preprocess the text inputs to the Roman alphabet. You can check if the tokenizer requires uroman as shown below.


  ```
  # pip install -U uroman
  from transformers import VitsTokenizer

  tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
  print(tokenizer.is_uroman)
  ```

  If your language requires uroman, the tokenizer automatically applies it to the text inputs. Python >= 3.10 doesn’t require any additional preprocessing steps. For Python < 3.10, follow the steps below.


  ```
  git clone https://github.com/isi-nlp/uroman.git
  cd uroman
  export UROMAN=$(pwd)
  ```

  Create a function to preprocess the inputs. You can either use the bash variable `UROMAN` or pass the directory path directly to the function.


  ```
  import torch
  from transformers import VitsTokenizer, VitsModel, set_seed
  import os
  import subprocess

  tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
  model = VitsModel.from_pretrained("facebook/mms-tts-kor")

  def uromanize(input_string, uroman_path):
      """Convert non-Roman strings to Roman using the `uroman` perl package."""
      script_path = os.path.join(uroman_path, "bin", "uroman.pl")

      command = ["perl", script_path]

      process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      # Execute the perl command
      stdout, stderr = process.communicate(input=input_string.encode())

      if process.returncode != 0:
          raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

      # Return the output as a string and skip the new-line character at the end
      return stdout.decode()[:-1]

  text = "이봐 무슨 일이야"
  uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

  inputs = tokenizer(text=uromanized_text, return_tensors="pt")

  set_seed(555)  # make deterministic
  with torch.no_grad():
     outputs = model(inputs["input_ids"])

  waveform = outputs.waveform[0]
  ```

## VitsConfig

### class transformers.VitsConfig

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/configuration_vits.py#L24)

( vocab\_size = 38 hidden\_size = 192 num\_hidden\_layers = 6 num\_attention\_heads = 2 window\_size = 4 use\_bias = True ffn\_dim = 768 layerdrop = 0.1 ffn\_kernel\_size = 3 flow\_size = 192 spectrogram\_bins = 513 hidden\_act = 'relu' hidden\_dropout = 0.1 attention\_dropout = 0.1 activation\_dropout = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 use\_stochastic\_duration\_prediction = True num\_speakers = 1 speaker\_embedding\_size = 0 upsample\_initial\_channel = 512 upsample\_rates = [8, 8, 2, 2] upsample\_kernel\_sizes = [16, 16, 4, 4] resblock\_kernel\_sizes = [3, 7, 11] resblock\_dilation\_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]] leaky\_relu\_slope = 0.1 depth\_separable\_channels = 2 depth\_separable\_num\_layers = 3 duration\_predictor\_flow\_bins = 10 duration\_predictor\_tail\_bound = 5.0 duration\_predictor\_kernel\_size = 3 duration\_predictor\_dropout = 0.5 duration\_predictor\_num\_flows = 4 duration\_predictor\_filter\_channels = 256 prior\_encoder\_num\_flows = 4 prior\_encoder\_num\_wavenet\_layers = 4 posterior\_encoder\_num\_wavenet\_layers = 16 wavenet\_kernel\_size = 5 wavenet\_dilation\_rate = 1 wavenet\_dropout = 0.0 speaking\_rate = 1.0 noise\_scale = 0.667 noise\_scale\_duration = 0.8 sampling\_rate = 16000 \*\*kwargs  )

Parameters

* **vocab\_size** (`int`, *optional*, defaults to 38) —
  Vocabulary size of the VITS model. Defines the number of different tokens that can be represented by the
  `inputs_ids` passed to the forward method of [VitsModel](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsModel).
* **hidden\_size** (`int`, *optional*, defaults to 192) —
  Dimensionality of the text encoder layers.
* **num\_hidden\_layers** (`int`, *optional*, defaults to 6) —
  Number of hidden layers in the Transformer encoder.
* **num\_attention\_heads** (`int`, *optional*, defaults to 2) —
  Number of attention heads for each attention layer in the Transformer encoder.
* **window\_size** (`int`, *optional*, defaults to 4) —
  Window size for the relative positional embeddings in the attention layers of the Transformer encoder.
* **use\_bias** (`bool`, *optional*, defaults to `True`) —
  Whether to use bias in the key, query, value projection layers in the Transformer encoder.
* **ffn\_dim** (`int`, *optional*, defaults to 768) —
  Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
* **layerdrop** (`float`, *optional*, defaults to 0.1) —
  The LayerDrop probability for the encoder. See the [LayerDrop paper](see <https://huggingface.co/papers/1909.11556>)
  for more details.
* **ffn\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.
* **flow\_size** (`int`, *optional*, defaults to 192) —
  Dimensionality of the flow layers.
* **spectrogram\_bins** (`int`, *optional*, defaults to 513) —
  Number of frequency bins in the target spectrogram.
* **hidden\_act** (`str` or `function`, *optional*, defaults to `"relu"`) —
  The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
  `"relu"`, `"selu"` and `"gelu_new"` are supported.
* **hidden\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout probability for all fully connected layers in the embeddings and encoder.
* **attention\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for the attention probabilities.
* **activation\_dropout** (`float`, *optional*, defaults to 0.1) —
  The dropout ratio for activations inside the fully connected layer.
* **initializer\_range** (`float`, *optional*, defaults to 0.02) —
  The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
* **layer\_norm\_eps** (`float`, *optional*, defaults to 1e-05) —
  The epsilon used by the layer normalization layers.
* **use\_stochastic\_duration\_prediction** (`bool`, *optional*, defaults to `True`) —
  Whether to use the stochastic duration prediction module or the regular duration predictor.
* **num\_speakers** (`int`, *optional*, defaults to 1) —
  Number of speakers if this is a multi-speaker model.
* **speaker\_embedding\_size** (`int`, *optional*, defaults to 0) —
  Number of channels used by the speaker embeddings. Is zero for single-speaker models.
* **upsample\_initial\_channel** (`int`, *optional*, defaults to 512) —
  The number of input channels into the HiFi-GAN upsampling network.
* **upsample\_rates** (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`) —
  A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network.
  The length of `upsample_rates` defines the number of convolutional layers and has to match the length of
  `upsample_kernel_sizes`.
* **upsample\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`) —
  A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling
  network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match
  the length of `upsample_rates`.
* **resblock\_kernel\_sizes** (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`) —
  A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN
  multi-receptive field fusion (MRF) module.
* **resblock\_dilation\_sizes** (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) —
  A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the
  HiFi-GAN multi-receptive field fusion (MRF) module.
* **leaky\_relu\_slope** (`float`, *optional*, defaults to 0.1) —
  The angle of the negative slope used by the leaky ReLU activation.
* **depth\_separable\_channels** (`int`, *optional*, defaults to 2) —
  Number of channels to use in each depth-separable block.
* **depth\_separable\_num\_layers** (`int`, *optional*, defaults to 3) —
  Number of convolutional layers to use in each depth-separable block.
* **duration\_predictor\_flow\_bins** (`int`, *optional*, defaults to 10) —
  Number of channels to map using the unonstrained rational spline in the duration predictor model.
* **duration\_predictor\_tail\_bound** (`float`, *optional*, defaults to 5.0) —
  Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor
  model.
* **duration\_predictor\_kernel\_size** (`int`, *optional*, defaults to 3) —
  Kernel size of the 1D convolution layers used in the duration predictor model.
* **duration\_predictor\_dropout** (`float`, *optional*, defaults to 0.5) —
  The dropout ratio for the duration predictor model.
* **duration\_predictor\_num\_flows** (`int`, *optional*, defaults to 4) —
  Number of flow stages used by the duration predictor model.
* **duration\_predictor\_filter\_channels** (`int`, *optional*, defaults to 256) —
  Number of channels for the convolution layers used in the duration predictor model.
* **prior\_encoder\_num\_flows** (`int`, *optional*, defaults to 4) —
  Number of flow stages used by the prior encoder flow model.
* **prior\_encoder\_num\_wavenet\_layers** (`int`, *optional*, defaults to 4) —
  Number of WaveNet layers used by the prior encoder flow model.
* **posterior\_encoder\_num\_wavenet\_layers** (`int`, *optional*, defaults to 16) —
  Number of WaveNet layers used by the posterior encoder model.
* **wavenet\_kernel\_size** (`int`, *optional*, defaults to 5) —
  Kernel size of the 1D convolution layers used in the WaveNet model.
* **wavenet\_dilation\_rate** (`int`, *optional*, defaults to 1) —
  Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.
* **wavenet\_dropout** (`float`, *optional*, defaults to 0.0) —
  The dropout ratio for the WaveNet layers.
* **speaking\_rate** (`float`, *optional*, defaults to 1.0) —
  Speaking rate. Larger values give faster synthesised speech.
* **noise\_scale** (`float`, *optional*, defaults to 0.667) —
  How random the speech prediction is. Larger values create more variation in the predicted speech.
* **noise\_scale\_duration** (`float`, *optional*, defaults to 0.8) —
  How random the duration prediction is. Larger values create more variation in the predicted durations.
* **sampling\_rate** (`int`, *optional*, defaults to 16000) —
  The sampling rate at which the output audio waveform is digitalized expressed in hertz (Hz).

This is the configuration class to store the configuration of a [VitsModel](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsModel). It is used to instantiate a VITS
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the VITS
[facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the
documentation from [PretrainedConfig](/docs/transformers/v4.56.2/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:


```
>>> from transformers import VitsModel, VitsConfig

>>> # Initializing a "facebook/mms-tts-eng" style configuration
>>> configuration = VitsConfig()

>>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
>>> model = VitsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VitsTokenizer

### class transformers.VitsTokenizer

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/tokenization_vits.py#L47)

( vocab\_file pad\_token = '<pad>' unk\_token = '<unk>' language = None add\_blank = True normalize = True phonemize = True is\_uroman = False \*\*kwargs  )

Parameters

* **vocab\_file** (`str`) —
  Path to the vocabulary file.
* **language** (`str`, *optional*) —
  Language identifier.
* **add\_blank** (`bool`, *optional*, defaults to `True`) —
  Whether to insert token id 0 in between the other tokens.
* **normalize** (`bool`, *optional*, defaults to `True`) —
  Whether to normalize the input text by removing all casing and punctuation.
* **phonemize** (`bool`, *optional*, defaults to `True`) —
  Whether to convert the input text into phonemes.
* **is\_uroman** (`bool`, *optional*, defaults to `False`) —
  Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.

Construct a VITS tokenizer. Also supports MMS-TTS.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.56.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### normalize\_text

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/tokenization_vits.py#L115)

( input\_string  )

Lowercase the input string, respecting any special token ids that may be part or entirely upper-cased.

#### prepare\_for\_tokenization

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/tokenization_vits.py#L142)

( text: str is\_split\_into\_words: bool = False normalize: typing.Optional[bool] = None \*\*kwargs  ) → `tuple[str, dict[str, Any]]`

Parameters

* **text** (`str`) —
  The text to prepare.
* **is\_split\_into\_words** (`bool`, *optional*, defaults to `False`) —
  Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
  tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
  which it will tokenize.
* **normalize** (`bool`, *optional*, defaults to `None`) —
  Whether or not to apply punctuation and casing normalization to the text inputs. Typically, VITS is
  trained on lower-cased and un-punctuated text. Hence, normalization is used to ensure that the input
  text consists only of lower-case characters.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Keyword arguments to use for the tokenization.

Returns

`tuple[str, dict[str, Any]]`

The prepared text and the unused kwargs.

Performs any necessary transformations before tokenization.

This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
`kwargs` at the end of the encoding process to be sure all the arguments have been used.

* **call**
* save\_vocabulary

## VitsModel

### class transformers.VitsModel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/modeling_vits.py#L1249)

( config: VitsConfig  )

Parameters

* **config** ([VitsConfig](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsConfig)) —
  Model configuration class with all the parameters of the model. Initializing with a config file does not
  load the weights associated with the model, only the configuration. Check out the
  [from\_pretrained()](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The complete VITS model, for text-to-speech synthesis.

This model inherits from [PreTrainedModel](/docs/transformers/v4.56.2/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/vits/modeling_vits.py#L1279)

( input\_ids: typing.Optional[torch.Tensor] = None attention\_mask: typing.Optional[torch.Tensor] = None speaker\_id: typing.Optional[int] = None output\_attentions: typing.Optional[bool] = None output\_hidden\_states: typing.Optional[bool] = None return\_dict: typing.Optional[bool] = None labels: typing.Optional[torch.FloatTensor] = None  ) → `transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)`

Parameters

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
* **speaker\_id** (`int`, *optional*) —
  Which speaker embedding to use. Only used for multispeaker models.
* **output\_attentions** (`bool`, *optional*) —
  Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
  tensors for more detail.
* **output\_hidden\_states** (`bool`, *optional*) —
  Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
  more detail.
* **return\_dict** (`bool`, *optional*) —
  Whether or not to return a [ModelOutput](/docs/transformers/v4.56.2/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
* **labels** (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*) —
  Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
  computation.

Returns

`transformers.models.vits.modeling_vits.VitsModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.vits.modeling_vits.VitsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VitsConfig](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsConfig)) and inputs.

* **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — The final audio waveform predicted by the model.
* **sequence\_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) — The length in samples of each element in the `waveform` batch.
* **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) — The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
  GAN decoder model to obtain the final audio waveform.
* **hidden\_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
* **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

The [VitsModel](/docs/transformers/v4.56.2/en/model_doc/vits#transformers.VitsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Example:


```
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

* forward

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/vits.md)
