# Hyperparameters
batch_size = 32
length_seconds = 2
seed = 1337 # Reproduciblity

train_ds = triple_ds(
  Path(data_dir).joinpath(train_name),
  batch_size=batch_size,
  sample_rate=sample_rate,
  length_samples=length_seconds*sample_rate,
  max_triplets_per_speaker=150,
  seed=seed
)

test_dual_ds = dual_ds(
  Path(data_dir).joinpath(test_name),
  batch_size=batch_size,
  sample_rate=sample_rate,
  length_samples=length_seconds*sample_rate,
  max_pairs_per_speaker=None,
  seed=seed
)

dev_ds = triple_ds(
  Path(data_dir).joinpath(dev_name),
  batch_size=batch_size,
  sample_rate=sample_rate,
  length_samples=length_seconds*sample_rate,
  max_triplets_per_speaker=None,
  seed=seed
)

dev_dual_ds = dual_ds(
  Path(data_dir).joinpath(dev_name),
  batch_size=batch_size,
  sample_rate=sample_rate,
  length_samples=length_seconds*sample_rate,
  max_pairs_per_speaker=None,
  seed=seed
)
