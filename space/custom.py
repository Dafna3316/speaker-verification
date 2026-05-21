import torch
import torchaudio
import torch.nn.functional as F

import keras
from keras import layers
from keras import ops

# Convert raw samples to a log-mel spectrogram
@keras.saving.register_keras_serializable()
class LogMelSpectrogram(layers.Layer):
    sample_rate: int
    fft_size: int
    step_size: int
    num_mel_bins: int

    # fft_size - bigger gets us more accurate frequencies but less accurate timing
    def __init__(self, sample_rate, fft_size=1024, step_size=128, num_mel_bins=64, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.step_size = step_size
        self.num_mel_bins = num_mel_bins

    def build(self, input_shape):
        # Compute the weights once
        mel_weights = torchaudio.functional.melscale_fbanks(
            n_freqs=self.fft_size // 2 + 1,
            f_min=20.0,
            f_max=self.sample_rate / 2, # Nyquist-Shannon
            n_mels=self.num_mel_bins,
            sample_rate=self.sample_rate,
            norm="slaney",
        )

        # Register the mel weights as a Keras non-trainable weight.
        # This ensures correct movement between CPU/GPU
        self.mel_weights = self.add_weight(
            name="mel_weights",
            shape=mel_weights.shape,
            initializer=keras.initializers.Constant(mel_weights.numpy()),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, batch):
        # Remove unnecessary channels
        if len(batch.shape) == 3 and batch.shape[-1] == 1:
            waveforms = ops.squeeze(batch, axis=-1)
        else: waveforms = batch

        # DFT
        stft = torch.stft(
            waveforms,
            n_fft=self.fft_size,
            hop_length=self.step_size,
            win_length=self.fft_size,
            window=torch.hann_window(self.fft_size, device=waveforms.device),
            center=False, # Don't pad end
            return_complex=True,
        )

        # Convert from (..., bins, time) to (..., time, bins)
        stft = ops.transpose(stft, axes=(0, 2, 1))

        # Strength of frequency is strictly nonnegative
        spectrograms = ops.abs(stft)
        mel_spectros = ops.tensordot(spectrograms, self.mel_weights, axes=1)
        # We don't want the log to be zero, so we add epsilon
        log_mels = ops.log(mel_spectros + keras.backend.epsilon())
        return ops.expand_dims(log_mels, axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        input_length = input_shape[1]

        # Very similar calculation for a convolution layer
        if input_length is None: num_frames = None
        else: num_frames = (input_length - self.fft_size) // self.step_size + 1

        return (batch_size, num_frames, self.num_mel_bins, 1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "step_size": self.step_size,
            "num_mel_bins": self.num_mel_bins,
        })
        return config

@keras.saving.register_keras_serializable()
class TripletLoss(keras.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true is always 0,
        # y_pred is (anchor, positive, negative)

        anchor = y_pred[:, 0, :]
        positive = y_pred[:, 1, :]
        negative = y_pred[:, 2, :]

        dist_pos = ops.sum(ops.square(anchor - positive), axis=1)
        dist_neg = ops.sum(ops.square(anchor - negative), axis=1)

        return ops.mean(ops.maximum(dist_pos - dist_neg + self.margin, 0.0))

    def get_config(self):
        config = super().get_config()
        config.update({
            "margin": self.margin
        })
        return config

@keras.saving.register_keras_serializable()
class L2Normalize(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, batch):
        return F.normalize(batch, p=2.0, dim=1)

    def compute_output_shape(self, input_shape):
        return input_shape

@keras.saving.register_keras_serializable()
class ResidualBlock2D(layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        # Correct deserialization
        self.strides = tuple(strides) if isinstance(strides, list) else strides

        self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="same")
        self.bn2 = layers.BatchNormalization()

        self.add = layers.Add()
        self.relu2 = layers.ReLU()

    def build(self, input_shape):
        channels = input_shape[-1]
        needs_projection = (channels != self.filters) or (self.strides != (1, 1) and self.strides != 1)
        if needs_projection:
            self.shortcut_conv = layers.Conv2D(self.filters, (1, 1), strides=self.strides, padding="same")
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv is None:
            shortcut = inputs
        else:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = self.add([x, shortcut])
        return self.relu2(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides
        })
        return config

@keras.saving.register_keras_serializable()
class ResidualBlock1D(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = layers.Conv1D(filters, kernel_size, strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv1D(filters, kernel_size, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.add = layers.Add()
        self.relu2 = layers.ReLU()

    def build(self, input_shape):
        channels = input_shape[-1]
        needs_projection = channels != self.filters or self.strides != 1
        if needs_projection:
            self.shortcut_conv = layers.Conv1D(self.filters, 1, strides=self.strides, padding="same")
            self.shortcut_bn = layers.BatchNormalization()
        else:
            self.shortcut_conv = None
            self.shortcut_bn = None

        super().build(input_shape)

    def call(self, inputs, training=False):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.shortcut_conv is None:
            shortcut = inputs
        else:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = self.add([x, shortcut])
        return self.relu2(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides
        })
        return config

@keras.saving.register_keras_serializable()
class MobilenetPreprocess(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, batch):
        return keras.applications.mobilenet_v2.preprocess_input(batch)