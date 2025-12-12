# Utilities for FeatureExtractors

This page lists all the utility functions that can be used by the audio `FeatureExtractor` in order to compute special features from a raw audio using common algorithms such as *Short Time Fourier Transform* or *log mel spectrogram*.

Most of those are only useful if you are studying the code of the audio processors in the library.

## Audio Transformations

#### transformers.audio\_utils.hertz\_to\_mel

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L248)

( freq: typing.Union[float, numpy.ndarray] mel\_scale: str = 'htk'  ) → `float` or `np.ndarray`

Parameters

* **freq** (`float` or `np.ndarray`) —
  The frequency, or multiple frequencies, in hertz (Hz).
* **mel\_scale** (`str`, *optional*, defaults to `"htk"`) —
  The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

Returns

`float` or `np.ndarray`

The frequencies on the mel scale.

Convert frequency from hertz to mels.

#### transformers.audio\_utils.mel\_to\_hertz

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L284)

( mels: typing.Union[float, numpy.ndarray] mel\_scale: str = 'htk'  ) → `float` or `np.ndarray`

Parameters

* **mels** (`float` or `np.ndarray`) —
  The frequency, or multiple frequencies, in mels.
* **mel\_scale** (`str`, *optional*, `"htk"`) —
  The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

Returns

`float` or `np.ndarray`

The frequencies in hertz.

Convert frequency from mels to hertz.

#### transformers.audio\_utils.mel\_filter\_bank

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L440)

( num\_frequency\_bins: int num\_mel\_filters: int min\_frequency: float max\_frequency: float sampling\_rate: int norm: typing.Optional[str] = None mel\_scale: str = 'htk' triangularize\_in\_mel\_space: bool = False  ) → `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`)

Parameters

* **num\_frequency\_bins** (`int`) —
  Number of frequency bins (should be the same as `n_fft // 2 + 1` where `n_fft` is the size of the Fourier Transform used to compute the spectrogram).
* **num\_mel\_filters** (`int`) —
  Number of mel filters to generate.
* **min\_frequency** (`float`) —
  Lowest frequency of interest in Hz.
* **max\_frequency** (`float`) —
  Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
* **sampling\_rate** (`int`) —
  Sample rate of the audio waveform.
* **norm** (`str`, *optional*) —
  If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
* **mel\_scale** (`str`, *optional*, defaults to `"htk"`) —
  The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
* **triangularize\_in\_mel\_space** (`bool`, *optional*, defaults to `False`) —
  If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
  should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

Returns

`np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`)

Triangular filter bank matrix. This is a
projection matrix to go from a spectrogram to a mel spectrogram.

Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

Different banks of mel filters were introduced in the literature. The following variations are supported:

* MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
  bandwidth of `[0, 4600]` Hz.
* MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
  bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
* MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
  speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
* HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
  12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio’s
`melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

#### transformers.audio\_utils.optimal\_fft\_length

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L534)

( window\_length: int  )

Finds the best FFT input size for a given `window_length`. This function takes a given window length and, if not
already a power of two, rounds it up to the next power or two.

The FFT algorithm works fastest when the length of the input is a power of two, which may be larger than the size
of the window or analysis frame. For example, if the window is 400 samples, using an FFT input size of 512 samples
is more optimal than an FFT size of 400 samples. Using a larger FFT size does not affect the detected frequencies,
it simply gives a higher frequency resolution (i.e. the frequency bins are smaller).

#### transformers.audio\_utils.window\_function

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L547)

( window\_length: int name: str = 'hann' periodic: bool = True frame\_length: typing.Optional[int] = None center: bool = True  )

Parameters

* **window\_length** (`int`) —
  The length of the window in samples.
* **name** (`str`, *optional*, defaults to `"hann"`) —
  The name of the window function.
* **periodic** (`bool`, *optional*, defaults to `True`) —
  Whether the window is periodic or symmetric.
* **frame\_length** (`int`, *optional*) —
  The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
  than the frame length, so that it will be zero-padded.
* **center** (`bool`, *optional*, defaults to `True`) —
  Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

Returns an array containing the specified window. This window is intended to be used with `stft`.

The following window types are supported:

* `"boxcar"`: a rectangular window
* `"hamming"`: the Hamming window
* `"hann"`: the Hann window
* `"povey"`: the Povey window

#### transformers.audio\_utils.spectrogram

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L611)

( waveform: ndarray window: ndarray frame\_length: int hop\_length: int fft\_length: typing.Optional[int] = None power: typing.Optional[float] = 1.0 center: bool = True pad\_mode: str = 'reflect' onesided: bool = True dither: float = 0.0 preemphasis: typing.Optional[float] = None mel\_filters: typing.Optional[numpy.ndarray] = None mel\_floor: float = 1e-10 log\_mel: typing.Optional[str] = None reference: float = 1.0 min\_value: float = 1e-10 db\_range: typing.Optional[float] = None remove\_dc\_offset: typing.Optional[bool] = None dtype: dtype = <class 'numpy.float32'>  )

Parameters

* **waveform** (`np.ndarray` of shape `(length,)`) —
  The input waveform. This must be a single real-valued, mono waveform.
* **window** (`np.ndarray` of shape `(frame_length,)`) —
  The windowing function to apply, including zero-padding if necessary. The actual window length may be
  shorter than `frame_length`, but we’re assuming the array has already been zero-padded.
* **frame\_length** (`int`) —
  The length of the analysis frames in samples. With librosa this is always equal to `fft_length` but we also
  allow smaller sizes.
* **hop\_length** (`int`) —
  The stride between successive analysis frames in samples.
* **fft\_length** (`int`, *optional*) —
  The size of the FFT buffer in samples. This determines how many frequency bins the spectrogram will have.
  For optimal speed, this should be a power of two. If `None`, uses `frame_length`.
* **power** (`float`, *optional*, defaults to 1.0) —
  If 1.0, returns the amplitude spectrogram. If 2.0, returns the power spectrogram. If `None`, returns
  complex numbers.
* **center** (`bool`, *optional*, defaults to `True`) —
  Whether to pad the waveform so that frame `t` is centered around time `t * hop_length`. If `False`, frame
  `t` will start at time `t * hop_length`.
* **pad\_mode** (`str`, *optional*, defaults to `"reflect"`) —
  Padding mode used when `center` is `True`. Possible values are: `"constant"` (pad with zeros), `"edge"`
  (pad with edge values), `"reflect"` (pads with mirrored values).
* **onesided** (`bool`, *optional*, defaults to `True`) —
  If True, only computes the positive frequencies and returns a spectrogram containing `fft_length // 2 + 1`
  frequency bins. If False, also computes the negative frequencies and returns `fft_length` frequency bins.
* **dither** (`float`, *optional*, defaults to 0.0) —
  Adds dithering. In other words, adds a small Gaussian noise to each frame.
  E.g. use 4.0 to add dithering with a normal distribution centered
  around 0.0 with standard deviation 4.0, 0.0 means no dithering.
  Dithering has similar effect as `mel_floor`. It reduces the high log\_mel\_fbank
  values for signals with hard-zero sections, when VAD cutoff is present in the signal.
* **preemphasis** (`float`, *optional*) —
  Coefficient for a low-pass filter that applies pre-emphasis before the DFT.
* **mel\_filters** (`np.ndarray` of shape `(num_freq_bins, num_mel_filters)`, *optional*) —
  The mel filter bank. If supplied, applies a this filter bank to create a mel spectrogram.
* **mel\_floor** (`float`, *optional*, defaults to 1e-10) —
  Minimum value of mel frequency banks.
* **log\_mel** (`str`, *optional*) —
  How to convert the spectrogram to log scale. Possible options are: `None` (don’t convert), `"log"` (take
  the natural logarithm) `"log10"` (take the base-10 logarithm), `"dB"` (convert to decibels). Can only be
  used when `power` is not `None`.
* **reference** (`float`, *optional*, defaults to 1.0) —
  Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
  the loudest part to 0 dB. Must be greater than zero.
* **min\_value** (`float`, *optional*, defaults to `1e-10`) —
  The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
  `log(0)`. For a power spectrogram, the default of `1e-10` corresponds to a minimum of -100 dB. For an
  amplitude spectrogram, the value `1e-5` corresponds to -100 dB. Must be greater than zero.
* **db\_range** (`float`, *optional*) —
  Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
  peak value and the smallest value will never be more than 80 dB. Must be greater than zero.
* **remove\_dc\_offset** (`bool`, *optional*) —
  Subtract mean from waveform on each frame, applied before pre-emphasis. This should be set to `true` in
  order to get the same results as `torchaudio.compliance.kaldi.fbank` when computing mel filters.
* **dtype** (`np.dtype`, *optional*, defaults to `np.float32`) —
  Data type of the spectrogram tensor. If `power` is None, this argument is ignored and the dtype will be
  `np.complex64`.

Calculates a spectrogram over one waveform using the Short-Time Fourier Transform.

This function can create the following kinds of spectrograms:

* amplitude spectrogram (`power = 1.0`)
* power spectrogram (`power = 2.0`)
* complex-valued spectrogram (`power = None`)
* log spectrogram (use `log_mel` argument)
* mel spectrogram (provide `mel_filters`)
* log-mel spectrogram (provide `mel_filters` and `log_mel`)

How this works:

1. The input waveform is split into frames of size `frame_length` that are partially overlapping by `frame\_length
   * hop\_length` samples.
2. Each frame is multiplied by the window and placed into a buffer of size `fft_length`.
3. The DFT is taken of each windowed frame.
4. The results are stacked into a spectrogram.

We make a distinction between the following “blocks” of sample data, each of which may have a different lengths:

* The analysis frame. This is the size of the time slices that the input waveform is split into.
* The window. Each analysis frame is multiplied by the window to avoid spectral leakage.
* The FFT input buffer. The length of this determines how many frequency bins are in the spectrogram.

In this implementation, the window is assumed to be zero-padded to have the same size as the analysis frame. A
padded window can be obtained from `window_function()`. The FFT input buffer may be larger than the analysis frame,
typically the next power of two.

Note: This function is not optimized for speed yet. It should be mostly compatible with `librosa.stft` and
`torchaudio.functional.transforms.Spectrogram`, although it is more flexible due to the different ways spectrograms
can be constructed.

#### transformers.audio\_utils.power\_to\_db

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L1033)

( spectrogram: ndarray reference: float = 1.0 min\_value: float = 1e-10 db\_range: typing.Optional[float] = None  ) → `np.ndarray`

Parameters

* **spectrogram** (`np.ndarray`) —
  The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
* **reference** (`float`, *optional*, defaults to 1.0) —
  Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
  the loudest part to 0 dB. Must be greater than zero.
* **min\_value** (`float`, *optional*, defaults to `1e-10`) —
  The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
  `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
* **db\_range** (`float`, *optional*) —
  Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
  peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

Returns

`np.ndarray`

the spectrogram in decibels

Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`, using basic
logarithm properties for numerical stability.

The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
This means that large variations in energy may not sound all that different if the sound is loud to begin with.
This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

Based on the implementation of `librosa.power_to_db`.

#### transformers.audio\_utils.amplitude\_to\_db

 [< source >](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/audio_utils.py#L1133)

( spectrogram: ndarray reference: float = 1.0 min\_value: float = 1e-05 db\_range: typing.Optional[float] = None  ) → `np.ndarray`

Parameters

* **spectrogram** (`np.ndarray`) —
  The input amplitude (mel) spectrogram.
* **reference** (`float`, *optional*, defaults to 1.0) —
  Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
  the loudest part to 0 dB. Must be greater than zero.
* **min\_value** (`float`, *optional*, defaults to `1e-5`) —
  The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
  `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
* **db\_range** (`float`, *optional*) —
  Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
  peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

Returns

`np.ndarray`

the spectrogram in decibels

Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
basic logarithm properties for numerical stability.

The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
This means that large variations in energy may not sound all that different if the sound is loud to begin with.
This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

[< > Update on GitHub](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/audio_utils.md)
