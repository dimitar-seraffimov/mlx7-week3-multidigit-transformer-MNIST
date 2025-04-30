import torch
import torch.nn as nn
from encoder import TransformerEncoder

#
#
# CONSTANTS
# 
#

PATCH_SIZE = 7 # 7x7 patches
NUM_PATCHES = 16 # 16 patches
EMBED_DIM = 32
NUM_CLASSES = 10

#
#
# PATCH EMBEDDING
#
#

class PatchEmbedding(nn.Module):
  def __init__(self, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM):
    super().__init__()
    self.linear = nn.Linear(patch_size * patch_size, embed_dim)

  def forward(self, x):
    """
    x: tensor of shape (batch_size, n_patches, patch_size * patch_size)
    """
    return self.linear(x)

#
#
# CLASSIFIER MODEL
#
#

class MNISTClassifier(nn.Module):
  def __init__(self, encoder: TransformerEncoder):
    super().__init__()
    self.patch_embedding = PatchEmbedding()
    self.encoder = encoder
    self.pos_embedding = nn.Parameter(torch.randn(1, NUM_PATCHES, EMBED_DIM))
    self.norm = nn.LayerNorm(EMBED_DIM)
    self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

  def forward(self, x):
    """
    x: tensor of shape (batch_size, n_patches, embed_dim)
    """
    x = self.patch_embedding(x)
    x = x + self.pos_embedding
    x = self.encoder(x) # shape: (batch_size, num_patches, embed_dim) -> 32 x 17 x 32
    x = x.mean(dim=1) # shape: (batch_size, embed_dim) -> 32 x 32 -> average over all patches
    x = self.norm(x) # normalise over the embedding dimension before classifier
    
    logits = self.classifier(x) # shape: (batch_size, num_classes) -> 32 x 10

    return logits
