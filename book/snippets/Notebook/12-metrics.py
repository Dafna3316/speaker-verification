def plot_history(history):
  fig, ax = plt.subplots(1)
  ax.plot(history.history["loss"], label="Train")
  ax.plot(history.history["val_loss"], label="Dev")
  ax.set_title("Loss")
  ax.set_ylabel("Loss")
  ax.set_xlabel("Epoch")
  ax.legend()
  plt.show()

def evaluate_accuracy(model, dual_ds):
  y_true = []
  dists = []

  # Collect true labels and distances
  for (chunk_a, chunk_b), labels in dual_ds:
    embedding_a = model.predict(chunk_a, verbose=0)
    embedding_b = model.predict(chunk_b, verbose=0)

    distances = np.sqrt(np.sum(np.square(embedding_a - embedding_b), axis=1))
    dists.extend(distances)
    y_true.extend(labels.numpy())

  y_true = np.array(y_true)
  dists = np.array(dists)

  # scikit-learn wants higher -> more similar, which is the opposite
  # of our sytsem.
  similarities = -dists

  # This gives us the FPR (=FAR) and TPR (=1-FRR) for various thresholds
  fpr, tpr, thresholds = roc_curve(y_true, similarities)
  # Coutn the area under the curve
  roc_auc = roc_auc_score(y_true, similarities)

  # FNR (=FRR)
  fnr = 1 - tpr

  # Find the threshold that minimizes the difference, i.e. brings FAR and FRR
  # closest together,
  eer_index = np.nanargmin(np.absolute(fnr - fpr))
  # its value,
  eer = fpr[eer_index]

  # and its location.
  optimal_threshold = -thresholds[eer_index]

  print(f"Area Under Curve (AUC): {roc_auc:.4f}")
  print(f"Equal Error Rate (EER): {eer:.4%}")
  print(f"Optimal Distance Threshold: {optimal_threshold:.4f}")

  return eer, roc_auc, optimal_threshold, (fpr, tpr)

def plot_roc_curve(fpr, tpr, roc_auc):
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
  plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
  plt.xlabel("False Acceptance Rate (FAR)")
  plt.ylabel("True Positive Rate (1 - FRR)")
  plt.title("Receiver Operating Characteristic")
  plt.legend(loc="lower right")
  plt.show()

def calculate_min_dcf(fpr, tpr, p_tar=0.01, c_miss=1.0, c_fa=1.0):
  fnr = 1 - tpr
  dcf_costs = (c_miss * fnr * p_tar) + (c_fa * fpr * (1 - p_tar))
  min_dcf = np.min(dcf_costs)
  c_default = min(c_miss * p_tar, c_fa * (1 - p_tar))
  mindcf_normalized = min_dcf / c_default
  print(f"minDCF: {mindcf_normalized:.4f}")
  return mindcf_normalized

def evaluate_on_set(model, ds, threshold, batch_limit=None):
  results = {
    "tp": [], # True positives
    "tn": [], # True negatives
    "fp": [], # False positives
    "fn": [], # False negatives
  }

  for idx, ((file_a, file_b), labels) in enumerate(ds):
    emb_a = model.predict(file_a, verbose=0)
    emb_b = model.predict(file_b, verbose=0)
    dists = np.sqrt(np.sum(np.square(emb_a - emb_b), axis=1))
    labels = labels.numpy()

    for i in range(len(labels)):
      dist = dists[i]
      y_true = labels[i]
      y_pred = 1 if dist < threshold else 0

      # Retrieve raw audio data
      audio_a = file_a[i].numpy()
      audio_b = file_b[i].numpy()

      record = {
        "a": audio_a,
        "b": audio_b,
        "distance": dist,
        "threshold": threshold
      }

      if y_true == 1 and y_pred == 1:
        results["tp"].append(record)
      elif y_true == 0 and y_pred == 0:
        results["tn"].append(record)
      elif y_true == 0 and y_pred == 1:
        results["fp"].append(record)
      elif y_true == 1 and y_pred == 0:
        results["fn"].append(record)

    if batch_limit is not None and idx >= batch_limit:
      break

  return results

# Output examples of wrongly classified pairs
def confusion_analysis(model, ds, threshold, batch_limit=None):
  results = evaluate_on_set(model, ds, threshold, batch_limit)

  if len(results["fp"]) > 0:
    samples = random.sample(results["fp"], 3)
    print("FALSE POSITIVES")
    for example in samples:
      print("A:")
      display(Audio(example["a"], rate=sample_rate))
      print("B:")
      display(Audio(example["b"], rate=sample_rate))
  if len(results["fn"]) > 0:
    samples = random.sample(results["fn"], 3)
    print("FALSE NEGATIVES")
    for example in samples:
      print("A:")
      display(Audio(example["a"], rate=sample_rate))
      print("B:")
      display(Audio(example["b"], rate=sample_rate))

  return results
