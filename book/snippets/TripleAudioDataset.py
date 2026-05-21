def decode_one_file_nocache(file, length_samples, sample_rate):
  # Decode the file
  audio, rate = torchaudio.load(file)

  # Resample if necessary
  if sample_rate is not None and rate != sample_rate:
    audio = torchaudio.functional.resample(audio, rate, sample_rate)

  # Downmix stereo
  if audio.shape[0] > 1: audio = torch.mean(audio, dim=0)
  # Remove the unnecessary channel dimension (in LibriSpeech always 1)
  else: audio = audio.squeeze()

  # Pad using repeat
  audio_length = audio.shape[0]

  if audio_length < length_samples:
    repeats = (length_samples // audio_length) + 1
    audio = audio.repeat(repeats)
    audio = audio[:length_samples]

  return audio

def read_and_crop_direct(audio, length_samples):
  # Random crop
  audio_length = audio.shape[0]
  if audio_length > length_samples:
    max_start = audio_length - length_samples
    start = torch.randint(0, max_start + 1, (1,)).item()
    audio = audio[start : start+length_samples]

  return audio

@functools.lru_cache(maxsize=1000)
def decode_one_file(file, length_samples, sample_rate):
  return decode_one_file_nocache(file, length_samples, sample_rate)

# Crop audio file to the correct length (randomly)
def read_and_crop(file, length_samples, sample_rate):
  audio = decode_one_file(file, length_samples, sample_rate)
  return read_and_crop_direct(audio, length_samples)

# Map file paths to actual audio dynamically
# the files are cropped randomly every epoch, acting as augmentation
class TripleAudioDataset(Dataset):
  def __init__(self, anchor, positive, negative, labels, sample_rate, length_samples):
    self.anchor = anchor
    self.positive = positive
    self.negative = negative
    self.labels = labels
    self.sample_rate = sample_rate
    self.length_samples = length_samples

  def __getitem__(self, idx):
    # Read and process the audio pairs
    audio_anchor = read_and_crop(self.anchor[idx], self.length_samples, self.sample_rate)
    audio_positive = read_and_crop(self.positive[idx], self.length_samples, self.sample_rate)
    audio_negative = read_and_crop(self.negative[idx], self.length_samples, self.sample_rate)

    # Return a tuple of inputs and the label
    return (audio_anchor, audio_positive, audio_negative), self.labels[idx]
