from IPython.display import Audio

for i in range(3):
  print(i)
  ((anchor, positive, negative), label) = train_ds.dataset[i]
  display(Audio(anchor, rate=sample_rate), Audio(positive, rate=sample_rate),
          Audio(negative, rate=sample_rate))

for _ in range(1):
  ((anchor, positive, negative), _) = train_ds.dataset[0]
  anchor = anchor.reshape((1,) + anchor.shape)
  positive = positive.reshape((1,) + positive.shape)
  negative = negative.reshape((1,) + negative.shape)
  spec_layer = LogMelSpectrogram(sample_rate=sample_rate)
  fig, axes = plt.subplots(3)
  axes[0].imshow(np.flip(spec_layer(anchor).cpu().numpy().squeeze().transpose()))
  axes[1].imshow(np.flip(spec_layer(positive).cpu().numpy().squeeze().transpose()))
  axes[2].imshow(np.flip(spec_layer(negative).cpu().numpy().squeeze().transpose()))
  fig.suptitle("Log-Mel Spectrograms of First Triplet")
  plt.show()
