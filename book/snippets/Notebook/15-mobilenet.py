# Preprocessing layer
@keras.saving.register_keras_serializable()
class MobilenetPreprocess(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, batch):
        return keras.applications.mobilenet_v2.preprocess_input(batch)

# Model definition
mobilenet = keras.Sequential([
    layers.Input(shape=audio_input_shape),
    LogMelSpectrogram(sample_rate=sample_rate),
    MobilenetPreprocess(),
    keras.applications.MobileNetV2(input_shape=spectro_shape[1:], include_top=False,
                                   weights=None, pooling="avg"),
    layers.Dropout(0.2),
    layers.Dense(2048),
    L2Normalize(),
])

mobilenet.summary()

# Training
siamese_mobilenet = siamese_from_model(mobilenet, audio_input_shape)
siamese_mobilenet.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=TripletLoss(margin=0.2))
history = siamese_mobilenet.fit(
    train_ds,
    epochs=30,
    callbacks=callbacks,
    validation_data=dev_ds,
)

# Metrics
plot_history(history)
eer, auc_score, optimal_margin, (fpr, tpr) = evaluate_accuracy(mobilenet, dev_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)

# Saving
mobilenet.save("mobilenet.keras")
display(FileLink("mobilenet.keras"))

# On test set
mobilenet = keras.saving.load_model("mobilenet.keras")
eer, auc_score, optimal_margin, (fpr, tpr) = evaluate_accuracy(mobilenet, test_dual_ds)
plot_roc_curve(fpr, tpr, auc_score)
