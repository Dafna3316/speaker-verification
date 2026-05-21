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
