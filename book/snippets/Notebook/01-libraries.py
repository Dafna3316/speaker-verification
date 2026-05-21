%load_ext autoreload
%autoreload 1

from pathlib import Path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import random

from sklearn.metrics import roc_curve, roc_auc_score
import gradio as gr

import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import keras
from keras import layers
from keras import ops

print(f"PyTorch {torch.__version__}")
print(f"Keras {keras.__version__} with backend {keras.backend.backend()}")
