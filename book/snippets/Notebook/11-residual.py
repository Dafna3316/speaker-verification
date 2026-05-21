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
