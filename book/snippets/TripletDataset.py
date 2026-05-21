# This will become the actual dataset
anchor_list = []
positive_list = []
negative_list = []

for speaker in speakers:
  files = speaker_to_files[speaker]
  positive_pairs = list(itertools.combinations(files, 2))
  random.shuffle(positive_pairs)

  # Truncate if necessary
  if max_triplets_per_speaker:
    positive_subset = positive_pairs[:max_triplets_per_speaker]
  else: positive_subset = positive_pairs

  # These are all positive pairs
  for anchor, positive in positive_subset:
    anchor_list.append(anchor)
    positive_list.append(positive)

    # Find a negative
    other_speaker = random.choice([s for s in speakers if s != speaker])
    negative = random.choice(speaker_to_files[other_speaker])
    negative_list.append(negative)

# Not actually used for anything
zero_labels = torch.zeros(len(anchor_list), dtype=torch.float32)

dataset = TripleAudioDataset(anchor_list, positive_list, negative_list,
                             zero_labels, sample_rate, length_samples)
