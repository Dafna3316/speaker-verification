%%writefile dataset.py
import os
import torch
import torchaudio
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import itertools
import functools

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

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    # Read and process the audio pairs
    audio_anchor = read_and_crop(self.anchor[idx], self.length_samples, self.sample_rate)
    audio_positive = read_and_crop(self.positive[idx], self.length_samples, self.sample_rate)
    audio_negative = read_and_crop(self.negative[idx], self.length_samples, self.sample_rate)

    # Return a tuple of inputs and the label
    return (audio_anchor, audio_positive, audio_negative), self.labels[idx]

# Traverse directory structure to generate dataset for a Siamese network with contrastive loss
def triple_ds(directory, length_samples, sample_rate=16000,
              batch_size=32, max_triplets_per_speaker=1000, seed=None):
  if seed:
    random.seed(seed)
    torch.manual_seed(seed)

  directory = Path(directory)

  # First-level directories correspond to speakers
  speaker_dirs = [d for d in directory.glob("*") if d.is_dir()]
  # Map of speaker name => file list
  speaker_to_files = {}

  # Populate speaker_to_files
  for d in speaker_dirs:
    files = [str(p) for p in d.glob("*.flac")]
    if len(files) >= 2:
      speaker_to_files[d.name] = files
    else:
      print("Warning: discarding speaker", d.name)

  if not speaker_to_files: raise ValueError("Not enough data")

  # Python crashes if I don't do this
  speakers = list(speaker_to_files.keys())

  # This will become the actual dataset
  anchor_list = []
  positive_list = []
  negative_list = []

  for speaker in speakers:
    files = speaker_to_files[speaker]
    positive_pairs = list(itertools.combinations(files, 2))
    random.shuffle(positive_pairs)

    # Truncate if necessary
    if max_triplets_per_speaker:
      positive_subset = positive_pairs[:max_triplets_per_speaker]
    else: positive_subset = positive_pairs

    # These are all positive pairs
    for anchor, positive in positive_subset:
      anchor_list.append(anchor)
      positive_list.append(positive)

      # Find a negative
      other_speaker = random.choice([s for s in speakers if s != speaker])
      negative = random.choice(speaker_to_files[other_speaker])
      negative_list.append(negative)

  # Not actually used for anything
  zero_labels = torch.zeros(len(anchor_list), dtype=torch.float32)

  dataset = TripleAudioDataset(anchor_list, positive_list, negative_list,
                               zero_labels, sample_rate, length_samples)
  loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(), # Compatibility with nVidia
  )

  return loader

class DualAudioDataset(Dataset):
  def __init__(self, path_a, path_b, labels, sample_rate, length_samples):
    self.path_a = path_a
    self.path_b = path_b
    self.labels = labels
    self.sample_rate = sample_rate
    self.length_samples = length_samples

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    # Read and process the audio pairs
    audio_a = read_and_crop(self.path_a[idx], self.length_samples, self.sample_rate)
    audio_b = read_and_crop(self.path_b[idx], self.length_samples, self.sample_rate)

    label = torch.tensor(self.labels[idx], dtype=torch.float32)

    # Return a tuple of inputs and the label
    return (audio_a, audio_b), label

def dual_ds(directory, length_samples, sample_rate=16000,
            batch_size=32, max_pairs_per_speaker=1000, seed=None):
  if seed:
    random.seed(seed)
    torch.manual_seed(seed)

  directory = Path(directory)

  # First-level directories correspond to speakers
  speaker_dirs = [d for d in directory.glob("*") if d.is_dir()]
  # Map of speaker name => file list
  speaker_to_files = {}

  # Populate speaker_to_files
  for d in speaker_dirs:
    files = [str(p) for p in d.glob("*.flac")]
    if len(files) >= 2:
      speaker_to_files[d.name] = files
    else:
      print("Warning: discarding speaker", d.name)

  if not speaker_to_files: raise ValueError("Not enough data")

  # Python crashes if I don't do this
  speakers = list(speaker_to_files.keys())

  # This will become the actual dataset
  a_list = []
  b_list = []
  label_list = []

  for speaker in speakers:
    files = speaker_to_files[speaker]
    positive_pairs = list(itertools.combinations(files, 2))
    random.shuffle(positive_pairs)

    # Truncate if necessary
    if max_pairs_per_speaker:
      positive_subset = positive_pairs[:max_pairs_per_speaker]
    else: positive_subset = positive_pairs

    # These are all positive pairs
    for a, b in positive_subset:
      a_list.append(a)
      b_list.append(b)
      label_list.append(1.0)

      # Find a negative
      other_speaker = random.choice([s for s in speakers if s != speaker])
      neg_a = random.choice(files)
      neg_b = random.choice(speaker_to_files[other_speaker])

      # Append the negative pair
      a_list.append(neg_a)
      b_list.append(neg_b)
      label_list.append(0.0)

  dataset = DualAudioDataset(a_list, b_list, label_list, sample_rate, length_samples)
  loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    prefetch_factor=2,
    pin_memory=torch.cuda.is_available(), # Compatibility with nVidia
  )

  return loader
