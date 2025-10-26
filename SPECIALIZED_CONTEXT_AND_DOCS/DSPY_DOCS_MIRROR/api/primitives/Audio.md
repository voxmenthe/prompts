# dspy.Audio

## dspy.Audio

```python
class Audio
```

### format

```python
def format(self)
```

### from_array

```python
def from_array(cls, array, sampling_rate, format='wav')
```

Process numpy-like array and encode it as base64. Uses sampling rate and audio format for encoding.


### from_file

```python
def from_file(cls, file_path)
```

Read local audio file and encode it as base64.


### from_url

```python
def from_url(cls, url)
```

Download an audio file from URL and encode it as base64.


### validate_input

```python
def validate_input(cls, values)
```

Validate input for Audio, expecting 'data' and 'audio_format' keys in dictionary.

Source: `/Volumes/cdrive/repos/OTHER_PEOPLES_REPOS/dspy/dspy/adapters/types/audio.py` (lines 20–112)

