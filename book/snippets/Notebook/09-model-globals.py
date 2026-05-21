callbacks = [
  keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                  start_from_epoch=5),
  keras.callbacks.ReduceLROnPlateau()
]

audio_input_shape=(length_seconds * sample_rate, 1)
spectro_shape = LogMelSpectrogram(sample_rate=sample_rate).compute_output_shape(
  (batch_size,) + audio_input_shape
)
