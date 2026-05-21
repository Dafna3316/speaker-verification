spect_2d = keras.Sequential([
  layers.Input(shape=audio_input_shape),
  LogMelSpectrogram(sample_rate=sample_rate),
  layers.Conv2D(16, 3, activation="relu", padding="same"),
  layers.Conv2D(16, 3, activation="relu", padding="same"),
  layers.BatchNormalization(),

  ResidualBlock2D(32),
  ResidualBlock2D(32, strides=2),
  ResidualBlock2D(32),
  ResidualBlock2D(32, strides=2),
  ResidualBlock2D(64),
  ResidualBlock2D(64, strides=2),
  ResidualBlock2D(64, strides=2),
  ResidualBlock2D(128),
  ResidualBlock2D(128, strides=2),
  ResidualBlock2D(128),
  ResidualBlock2D(256, strides=2),
  ResidualBlock2D(512, strides=2),

  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(2048, activation="relu"),
  L2Normalize(),
], name="cnn_2d_spectro")

spect_2d.summary()

# Training
siamese_spect_2d = siamese_from_model(spect_2d, audio_input_shape)
siamese_spect_2d.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=TripletLoss(margin=0.2))
history = siamese_spect_2d.fit(
  train_ds,
  epochs=30,
  callbacks=callbacks,
  validation_data=dev_ds,
)

# Metrics
plot_history(history)
eer, auc_score, optimal_threshold, (fpr, tpr) = evaluate_accuracy(spect_2d, dev_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)

# Saving
spect_2d.save("spect_2d.keras")
display(FileLink("spect_2d.keras"))

# On test set
spect_2d = keras.saving.load_model("spect_2d.keras")
eer, auc_score, optimal_threshold, (fpr, tpr) = evaluate_accuracy(spect_2d, test_dual_ds)
mindcf = calculate_min_dcf(fpr, tpr)
plot_roc_curve(fpr, tpr, auc_score)
_ = confusion_analysis(spect_2d, test_dual_ds, optimal_threshold, 1)
