audio_cnn = keras.Sequential([
  layers.Input(shape=audio_input_shape),
  layers.Conv1D(32, sample_rate // 8, strides=sample_rate // 600, activation="relu"),
  layers.Conv1D(32, 5, activation="relu", padding="same"),
  layers.BatchNormalization(),

  ResidualBlock1D(32, strides=2),
  ResidualBlock1D(64),
  ResidualBlock1D(64, strides=2),
  ResidualBlock1D(64, strides=2),
  ResidualBlock1D(64),
  ResidualBlock1D(128, strides=2),
  ResidualBlock1D(128),
  ResidualBlock1D(128, strides=2),
  ResidualBlock1D(256, strides=2),
  ResidualBlock1D(256),
  ResidualBlock1D(256, strides=2),
  ResidualBlock1D(512, strides=2),
  ResidualBlock1D(512),
  ResidualBlock1D(512, strides=2),

  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(2048, activation="relu"),
  L2Normalize(),
], name="cnn_1d_on_audio")

audio_cnn.summary()

# Training
siamese_audio_cnn = siamese_from_model(audio_cnn, audio_input_shape)

siamese_audio_cnn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=TripletLoss(margin=0.2))
history = siamese_audio_cnn.fit(
  train_ds,
  epochs=30,
  callbacks=callbacks,
  validation_data=dev_ds,
)

# History and metrics
plot_history(history)
eer, auc_score, optimal_margin, (fpr, tpr) = evaluate_accuracy(audio_cnn, dev_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)

# Save and download
from IPython.display import FileLink

audio_cnn.save("audio_cnn.keras")
display(FileLink("audio_cnn.keras"))

# On test set
audio_cnn = keras.saving.load_model("audio_cnn.keras")
eer, auc_score, optimal_margin, (fpr, tpr) = evaluate_accuracy(audio_cnn, test_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)
