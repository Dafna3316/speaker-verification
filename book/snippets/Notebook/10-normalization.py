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
