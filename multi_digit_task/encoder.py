import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# CONFIG
#
#

BATCH_SIZE = 32 # number of images in each training batch, sanity check = 32 * 1875 = 60000
NUM_LAYERS = 6 # number of encoder blocks
NUM_HEADS = 4 # number of attention heads
EMBED_DIM = 64
HIDDEN_DIM = 24
NUM_PATCHES = 16 # 16 patches

#
#
# ENCODER BLOCK
#
#

class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, hidden_dim, num_heads):
    """
    Encoder block with multi-head attention layers.
    - Input/Output shape: (N_PATCHES, EMBED_DIM)
    - Q, K, V projected to hidden_dim (24)
    - Final feed-forward projects back to embed_dim (64)
    """

    super().__init__()
    assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads

    # Q, K, V projection (64 -> 24)
    self.q_proj = nn.Linear(embed_dim, hidden_dim)
    self.k_proj = nn.Linear(embed_dim, hidden_dim)
    self.v_proj = nn.Linear(embed_dim, hidden_dim)

    # final linear layer after attention (24 -> 64), return back to embedding dimension
    self.out_proj = nn.Linear(hidden_dim, embed_dim)

    # feed-forward network 
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, embed_dim)
    )

    # layer norm
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    """
    x: tensor of shape (batch_size, n_patches, embed_dim)
    """
    batch_size, n_patches, _ = x.shape

    # linear projections
    Q = self.q_proj(x).view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, num_patches, head_dim)
    K = self.k_proj(x).view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1,2)
    V = self.v_proj(x).view(batch_size, n_patches, self.num_heads, self.head_dim).transpose(1,2)

    # attention scores
    attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5) # shape: (batch_size, num_heads, num_patches, num_patches) -> each element compares itself to every other element
    attn_weights = F.softmax(attn_score, dim=-1)

    # apply attention weights to V
    attn_output = torch.matmul(attn_weights, V) # shape: (batch_size, num_heads, num_patches, head_dim)

    # concatenate heads
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, n_patches, self.hidden_dim) # shape: (batch_size, num_patches, hidden_dim)

    # project back to embedding dimension
    out = self.out_proj(attn_output) # shape: (batch_size, n_patches, embed_dim)

    # residual connection + layer norm
    x = self.norm1(x + out)

    # feed-forward + residual + norm
    x = self.norm2(x + self.ffn(x))

    return x

#
#
# ENCODER STACK
#
#

class TransformerEncoder(nn.Module):
  def __init__(self, num_layers, embed_dim, hidden_dim, num_heads):
    """
    Transformer encoder stack with multiple encoder blocks.
    - Input/Output shape: (batch_size, n_patches, embed_dim)
    - num_layers: number of encoder blocks
    """

    super().__init__()

    # learnable positional encodings for each patch
    self.positional_encoding = nn.Parameter(torch.randn(1, NUM_PATCHES, EMBED_DIM)) 

    self.layers = nn.ModuleList(
      EncoderBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)
    )

  def forward(self, x):
    """
    x: tensor of shape (batch_size, n_patches, embed_dim)
    return: same shape as input
    """

    # positional encodings to each patch
    x = x + self.positional_encoding

    for layer in self.layers:
      x = layer(x)

    return x

#
#
# MAIN
#
#

if __name__ == "__main__":
  encoder = TransformerEncoder(num_layers=NUM_LAYERS, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS)
  
  test_input = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM)
  output = encoder(test_input)

  print("Output shape:", output.shape) # (batch_size, n_patches, embed_dim) -> 32 x 16 x 64
