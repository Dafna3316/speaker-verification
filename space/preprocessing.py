import torch
import torchaudio
import random

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