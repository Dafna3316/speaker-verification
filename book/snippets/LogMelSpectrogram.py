class LogMelSpectrogram(layers.Layer):
  sample_rate: int
  fft_size: int
  step_size: int
  num_mel_bins: int

  def build(self, input_shape):
    # Compute the weights once
    mel_weights = torchaudio.functional.melscale_fbanks(
      n_freqs=self.fft_size // 2 + 1,
      f_min=20.0,
      f_max=7600.0,
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
    log_mels = ops.log(mel_spectros + keras.backend.epsilon())
    return ops.expand_dims(log_mels, axis=-1)
