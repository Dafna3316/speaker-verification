# Create a siamese network from a given model
def siamese_from_model(model, input_shape):
  anchor_input = layers.Input(shape=input_shape, name="anchor")
  positive_input = layers.Input(shape=input_shape, name="positive")
  negative_input = layers.Input(shape=input_shape, name="negative")

  embedding_a = model(anchor_input)
  embedding_p = model(positive_input)
  embedding_n = model(negative_input)

  output = ops.stack([embedding_a, embedding_p, embedding_n], axis=1)

  return keras.Model(inputs=[anchor_input, positive_input, negative_input],
                     outputs=output, name="siamese_" + model.name)

def final_predictor_from_model(model, input_shape, margin):
  input_a = layers.Input(shape=input_shape, name="a")
  input_b = layers.Input(shape=input_shape, name="b")

  embedding_a = model(input_a)
  embedding_b = model(input_b)

  dist = ops.sqrt(ops.sum(ops.square(embedding_a - embedding_b), axis=1))
  prediction = (dist < margin).cast("float32")

  return keras.Model(inputs=[input_a, input_b],
                     outputs=prediction, name="predictor_" + model.name)

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
