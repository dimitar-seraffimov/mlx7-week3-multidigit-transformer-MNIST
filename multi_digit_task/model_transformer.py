import torch
import torch.nn as nn
from encoder import TransformerEncoder
from decoder import TransformerDecoder

#
#
# CONFIG
#
#

EMBED_DIM = 64 # embedding dimension
HIDDEN_DIM = 24 # hidden dimension
NUM_LAYERS = 6 # number of encoder blocks
NUM_HEADS = 4 # number of attention heads
NUM_PATCHES = 16 # number of patches = 4x4 grid of 14x14 patches in 56x56 image 
VOCAB_SIZE = 13 # vocabulary size = 0-9 and <sos>, <eos>, <empty>
MAX_SEQ_LEN = 5 # maximum sequence length = <sos> + 4 digits (<eos> is not passed to the decoder) 
PATCH_SIZE = 14 # patch size = 14x14 pixels
BATCH_SIZE = 32 # batch size for loading training data

#
#
# TRANSFORMER ENCODER-DECODER MODEL
#
#

class MultiDigitTransformer(nn.Module):
  def __init__(self):
    super().__init__()

    self.patch_embed = nn.Linear(PATCH_SIZE * PATCH_SIZE, EMBED_DIM)
    self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES, EMBED_DIM))

    #
    # ENCODER
    #
    self.encoder = TransformerEncoder(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS
    )

    #
    # DECODER
    #

    self.decoder = TransformerDecoder(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

  def forward(self, patch_inputs, decoder_input_tokens):
    """
    image_patches: (batch_size, num_patches, embed_dim) -> 32 x 16 x 64 - raw flattened+embedded image patches
    decoder_input_tokens: (batch_size, max_seq_len) -> 32 x 5 - token IDs for decoder (<sos> + digits)
    return: logits of shape (batch_size, max_seq_len, vocab_size) -> 32 x 5 x 13
    """

    embedded_patches = self.patch_embed(patch_inputs) # (batch_size, num_patches, embed_dim) -> 32 x 16 x 64
    embedded_patches = embedded_patches + self.pos_embed # (batch_size, num_patches, embed_dim) -> 32 x 16 x 64 - added positional embeddings

    encoder_output = self.encoder(embedded_patches)  # (batch_size, num_patches, embed_dim) -> 32 x 16 x 64
    decoder_output = self.decoder(decoder_input_tokens, encoder_output)  # (batch_size, max_seq_len, vocab_size) -> 32 x 5 x 13
    
    return decoder_output

#
#
# MAIN
#
#

if __name__ == "__main__":
    model = MultiDigitTransformer()

    patch_input = torch.randn(BATCH_SIZE, NUM_PATCHES, PATCH_SIZE * PATCH_SIZE)
    token_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

    output = model(patch_input, token_input)
    print("Model output shape:", output.shape)  # (batch_size, max_seq_len, vocab_size) -> 32 x 5 x 13
