import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import os
from pathlib import Path
import requests
import pandas as pd
import io
from tqdm import tqdm

#
#
# CONSTANTS
#
#

PATCH_SIZE = 14 # 14x14 patches
N_PATCHES = 16 # 4x4 grid of 14x14 in 56x56 image
DATA_DIR = "data"
HF_PARQUET_URLS = {
  "train": "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet",
  "test": "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet"
}

label_to_index = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    "<sos>": 10,
    "<eos>": 11,
    "<empty>": 12,
}

index_to_label = {v: k for k, v in label_to_index.items()}

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
# ADDITIONAL PREPROCESSING FUNCTIONS
#
#

# create empty (black) 28×28 images
def create_black_image():
    return Image.new('L', (28, 28), color=0) # 'L' stands for luminance, which is a grayscale image with only one channel

def load_mnist_images(split):
    parquet_file = download_mnist(split)
    df = pd.read_parquet(parquet_file)
    images = [Image.open(io.BytesIO(img["bytes"])).convert("L") for img in df["image"]]
    labels = df["label"].tolist()
    return images, labels

def create_mixed_image(images):
    # randomly select and combine four digit images into a single 56×56 image
    canvas = Image.new('L', (56, 56), color=0)
    positions = [(0, 0), (28, 0), (0, 28), (28, 28)] # [top-left, top-right, bottom-left, bottom-right]
    for pos, img in zip(positions, images):  # use passed-in imgs in order
      canvas.paste(img, pos)
    return canvas

#
#
# SAVE MULTI-DIGIT IMAGE DATASET
#
#

def save_multidigit_dataset(dataset, split, num_samples):
  print("[INFO] Saving multi-digit dataset...")
  records = []

  for i in tqdm(range(num_samples)):
    patches_tensor, label_tensor = dataset[i]

    record = {
      "patches": patches_tensor.tolist(),
      "label": label_tensor.tolist(),
    }
    records.append(record)

  df = pd.DataFrame(records)
  df.to_parquet(f"{DATA_DIR}/multidigit_{split}.parquet")
  print(f"[DONE] Multi-digit {split} dataset saved to {DATA_DIR}/multidigit_{split}.parquet")

#
#
# MULTI-DIGIT DATASET
#
#

class MultiDigitDataset(Dataset):
  """
  This dataset:
    - splits each 56x56 image into 16 patches of size 14x14
    - flattens each patch to a 1x196 and embeds each patch into 1x64 using a nn.Linear(196, 64) regression layer
    - returns a torch.Tensor of shape (16, 64) for each image
  """
  def __init__(self, split="train"):
    print("[INFO] Loading MNIST images and labels...")
    self.images, self.labels = load_mnist_images(split)
    self.transform = transforms.ToTensor()

  def __len__(self):
    return len(self.images) // 4 # 4 digit images per 56x56 image
  
  def __getitem__(self, idx):
    imgs = []
    label_ids = [] # [top-left, top-right, bottom-left, bottom-right]

    for i in range(4):
       if random.random() < 0.65: # 65% chance to be an empty image
          imgs.append(create_black_image())
          label_ids.append(label_to_index["<empty>"])
       else:
          idx = random.randint(0, len(self.images) - 1)
          imgs.append(self.images[idx])
          label_ids.append(label_to_index[self.labels[idx]])
    
    composed_img = create_mixed_image(imgs)
    img_tensor = self.transform(composed_img).squeeze(0) # remove the batch dimension

    patches = img_tensor.unfold(0, PATCH_SIZE, PATCH_SIZE).unfold(1, PATCH_SIZE, PATCH_SIZE)
    patches = patches.contiguous().view(-1, PATCH_SIZE * PATCH_SIZE) # flatten each patch into a vector of size 196, shape: (16, 196)

    # label encoding
    label_seq = [label_to_index["<sos>"]] + label_ids + [label_to_index["<eos>"]]
    label_tensor = torch.tensor(label_seq, dtype=torch.long)

    return patches, label_tensor

#
#
# MAIN
#
#

if __name__ == "__main__":
  from matplotlib import pyplot as plt
  train_dataset = MultiDigitDataset(split="train")
  test_dataset = MultiDigitDataset(split="test")

  save_multidigit_dataset(train_dataset, "train", 60000)
  save_multidigit_dataset(test_dataset, "test", 20000)
