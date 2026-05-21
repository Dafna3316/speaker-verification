# Path where the dataset is stored
data_dir = "/Users/tv/librispeech" ## CHANGE THIS

# See the website linked below for available splits
train_name = "train-clean-100"
test_name = "test-clean"
dev_name = "dev-clean"

# DON'T CHANGE
sample_rate = 16000 # DON'T CHANGE

# Check dependencies
import shutil
for requirement in ["curl", "tar", "grep", "ffmpeg"]:
  if not shutil.which(requirement):
    raise f"Dependency {requirement} not found!"

def ensure_set(name: str):
  """Ensures that a split of LibriSpeech is available."""

  basepath = Path(data_dir)
  # Expected directory for this split
  set_dir = basepath.joinpath(name)
  # Archive path
  tar = basepath.joinpath(name + ".tar.gz")
  # Ensure the main data directory exists
  basepath.mkdir(exist_ok=True, parents=True)

  if set_dir.exists():
    print(f"Set {name} already found")
    return

  # Download provided archive from the web using cURL
  if not tar.exists():
    print(f"Downloading LibriSpeech {name}...")
    url = "https://openslr.trmal.net/resources/12/" + name + ".tar.gz"
    !curl -Lo {tar.absolute()} {url}

  # Extract downloaded archive
  print(f"Extracting LibriSpeech {name}")
  set_dir.mkdir()
  # This grep pattern extracts the first-level directories after the split name,
  # which correspond to speaker IDs.
  # I❤️IPython
  speakers = !tar -tf {tar.absolute()} | grep '.*/'{name}'/[[:digit:]]*/$'
  for speaker in speakers:
    # Path to the speaker *within the archive*
    sp_in_tar = Path(speaker)
    # Path to the speaker directory
    speaker_dir = set_dir.joinpath(sp_in_tar.name)
    speaker_dir.mkdir()

    print(f"Extracting speaker {sp_in_tar.name}")
    # This pattern extracts all .flac files from the speaker directory
    # --strip-components 4 is used to remove leading directories (chapter information etc.)
    # as we already make sure the files are extracted to the speaker directory using -C
    !tar -C {speaker_dir.absolute()} --strip-components 4 -xzf {tar.absolute()} {sp_in_tar}'/**/*.flac'

  # Remove archive
  tar.unlink()

print("Downloading dataset (LibriSpeech)")
ensure_set(train_name)
ensure_set(dev_name)
ensure_set(test_name)
print("Done!")
