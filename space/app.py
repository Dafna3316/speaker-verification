import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torchaudio
import torch.nn.functional as F

import keras
from keras import layers
from keras import ops

from custom import LogMelSpectrogram, TripletLoss, L2Normalize, ResidualBlock2D, ResidualBlock1D
from preprocessing import decode_one_file_nocache, read_and_crop_direct
from config import sample_rate, length_samples, margins

import gradio as gr

models = {}

def get_model(name):
    if name not in models:
        models[name] = keras.saving.load_model(f"bin/{name}.keras")
    return models[name]

def model_predict(model, file_a, file_b):
    audio_a = decode_one_file_nocache(file_a, length_samples, sample_rate)
    audio_a = read_and_crop_direct(audio_a, length_samples)
    audio_a = audio_a.reshape((1,) + audio_a.shape)

    audio_b = decode_one_file_nocache(file_b, length_samples, sample_rate)
    audio_b = read_and_crop_direct(audio_b, length_samples)
    audio_b = audio_b.reshape((1,) + audio_b.shape)

    emb_a = model(audio_a, training=False).squeeze()
    emb_b = model(audio_b, training=False).squeeze()

    dist = ops.sqrt(ops.sum(ops.square(embedding_a - embedding_b)))

    return dist

def predict(name: str, file_a, file_b):
    assert name in margins
    margin = margins[name]
    dist = model_predict(get_model(name), file_a, file_b)

    if dist < margin:
        return f"""
          ✅ **Same Speaker** (was {dist:.2f} apart)
        """
    else:
        return f"""
          ❌ **Different Speaker** (was {dist:.2f} apart)
        """

model_dropdown = gr.Dropdown(choices=[
    ("1D CNN on Raw Audio", "audio_cnn"),
    ("1D CNN on Spectrogram", "spect_1d"),
    ("CNN Based on MobileNetV2", "mobilenet"),
    ("Original 2D CNN on Spectrogram", "spect_2d"),
], label="Model")

demo = gr.Interface(fn=predict, inputs=[
    model_dropdown,
    gr.Audio(type="filepath", label="First Recording"),
    gr.Audio(type="filepath", label="Second Recording"),
], outputs=gr.Markdown())

if __name__ == "__main__":
    demo.launch()
