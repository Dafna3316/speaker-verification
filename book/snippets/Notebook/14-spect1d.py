spect_1d = keras.Sequential([
  layers.Input(shape=audio_input_shape),
  LogMelSpectrogram(sample_rate=sample_rate),
  # The width dimension represents time, and the height represents frequencies.
  # We treat the frequencies as "features", so we can convolve over them.
  layers.Reshape((spectro_shape[1], spectro_shape[2])),
  layers.Conv1D(spectro_shape[2], 3, activation="relu", padding="same"),
  layers.Conv1D(64, 3, activation="relu", padding="same"),
  layers.BatchNormalization(),

  ResidualBlock1D(64),
  ResidualBlock1D(64, strides=2),
  ResidualBlock1D(128),
  ResidualBlock1D(128, strides=2),
  ResidualBlock1D(256),
  ResidualBlock1D(256, strides=2),
  ResidualBlock1D(256, strides=2),
  ResidualBlock1D(512),
  ResidualBlock1D(512, strides=2),
  ResidualBlock1D(512),
  ResidualBlock1D(512, strides=2),

  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(2048, activation="relu"),
  L2Normalize(),
], name="cnn_1d_spectro")

spect_1d.summary()

# Training
siamese_spect_1d = siamese_from_model(spect_1d, audio_input_shape)

siamese_spect_1d.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=TripletLoss(margin=0.2))
history = siamese_spect_1d.fit(
  train_ds,
  epochs=30,
  callbacks=callbacks,
  validation_data=dev_ds,
)

# Metrics
plot_history(history)
eer, auc_score, optimal_threshold, (fpr, tpr) = evaluate_accuracy(spect_1d, dev_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)

# Saving
spect_1d.save("spect_1d.keras")
display(FileLink("spect_1d.keras"))

# On test set
spect_1d = keras.saving.load_model("spect_1d.keras")
eer, auc_score, optimal_threshold, (fpr, tpr) = evaluate_accuracy(spect_1d, test_dual_ds)
mindcf = calculate_min_dcf(fpr, tpr)
plot_roc_curve(fpr, tpr, auc_score)
_ = confusion_analysis(spect_1d, test_dual_ds, optimal_threshold, 1)
