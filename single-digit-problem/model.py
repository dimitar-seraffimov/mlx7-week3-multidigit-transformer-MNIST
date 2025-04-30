import torch.nn as nn
from encoder import TransformerEncoder

#
#
# CONSTANTS
# 
#

EMBED_DIM = 32
NUM_CLASSES = 10

#
#
# CLASSIFIER MODEL
#
#

class MNISTClassifier(nn.Module):
  def __init__(self, encoder: TransformerEncoder):
    super().__init__()
    self.encoder = encoder
    self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

  def forward(self, x):
    """
    x: tensor of shape (batch_size, n_patches, embed_dim)
    """
    
    x = self.encoder(x) # shape: (batch_size, num_patches, embed_dim) -> 32 x 17 x 32
    x = x[:, 0] # select the [CLS] token, shape: (batch_size, embed_dim) -> 32 x 32
    logits = self.classifier(x) # shape: (batch_size, num_classes) -> 32 x 10

    return logits
