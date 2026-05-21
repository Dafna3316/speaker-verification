callbacks = [
  keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, start_from_epoch=5),
  keras.callbacks.ReduceLROnPlateau()
]

# $\ldots$

siamese_spect_1d = siamese_from_model(spect_1d, audio_input_shape)

siamese_spect_1d.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                         loss=TripletLoss(margin=0.2))
history = siamese_spect_1d.fit(
  train_ds,
  epochs=30,
  callbacks=callbacks,
  validation_data=dev_ds,
)
