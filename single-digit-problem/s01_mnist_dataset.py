import os
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import io
import requests
from torchvision.transforms import ToTensor

#
#
# CONSTANTS
#
#

PATCH_SIZE = 7 # 7x7 patches
EMBED_DIM = 32 # output size of the linear layer
N_PATCHES = 16 # 4x4 grid of 7x7 in 28x28 image
DATA_DIR = "data"
HF_PARQUET_URLS = {
  "train": "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet",
  "test": "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"
}

#
#
# DOWNLOAD $ SAVE MNIST DATASETS
#
#

def download_mnist(split):
  os.makedirs(DATA_DIR, exist_ok=True)
  path = Path(f"{DATA_DIR}/mnist_{split}.parquet")
  if not path.exists():
    print(f"[DOWNLOAD] Downloading MNIST {split} dataset to {path}")
    url = HF_PARQUET_URLS[split]
    r = requests.get(url)
    path.write_bytes(r.content)
    print(f"[DONE] MNIST {split} dataset downloaded to {path}")
  else:
    print(f"[SKIP] MNIST {split} dataset already exists in {path}")

  return path

#
#
# MNIST CLASS
#
#

class MNISTDataset(torch.utils.data.Dataset):
  """
  This dataset:
    - splits each 28x28 image into 16 patches of size 7x7
    - flattens each patch to a 1x49 and embeds each patch into 1x32 using a nn.Linear(49, 32) regression layer
    - adds random positional encodings to each patch
    - returns a torch.Tensor of shape (16, 32) for each image
  """

  def __init__(self, split="train"):
    # load the dataset
    parquet_file = download_mnist(split)
    df = pd.read_parquet(parquet_file)
    
    self.images = df["image"]
    self.labels = df["label"].tolist()

  def __len__(self):
    return len(self.images)
    
  def __getitem__(self, idx):
    # decode image
    img = Image.open(io.BytesIO(self.images[idx]["bytes"])).convert("L")
    img_tensor = ToTensor()(img).squeeze() # convert to tensor 28x28

    # split into patches
    patches = img_tensor.unfold(0, PATCH_SIZE, PATCH_SIZE).unfold(1, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE) # flatten each patch into a vector of size 49, shape: (16, 49)

    return patches, self.labels[idx]

#
#
# MAIN
#
#

if __name__ == "__main__":
  dataset = MNISTDataset(split="train")
  x, y = dataset[0]
  print(f"Embedded shape: {x.shape}, Label: {y}") # (16, 32), Label: int