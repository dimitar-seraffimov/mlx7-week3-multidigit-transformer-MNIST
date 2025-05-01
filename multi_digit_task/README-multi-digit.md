The files are listed in their order of creation:
s01_multi_digit_dataset.py

    Internal functionality:
      - download and load the MNIST dataset
      - create black (empty) 28×28 images for <empty> class
      - randomly select and combine four digit images into a single 56×56 image
      - divide the combined image into 16 patches of size 14×14
      - flatten each patch into a 1×196 vector
      - return raw patch tensors (16, 196) and the corresponding target sequence [<sos>, digit1, digit2, ..., <eos>]

encoder.py:

    Internal functionality:
    - define a multi-head self-attention encoder block (with nn.Linear projections for Q, K, V)
    for each head:
        - compute attention scores using scaled dot-product (Q × Kᵀ / √d)
        - apply softmax to get attention weights
        - compute weighted sum of V values using the attention scores
    - concatenate outputs from all attention heads
    - apply a final linear projection back to embedding size (24 → 64)
    - add a feed-forward network (MLP) with ReLu activation
    - use residual connections and layer normalization after both attention and FFN
    - stack multiple encoder blocks (in this example code -> 6 layers) using nn.ModuleList
    - accepts input of shape (batch_size, num_patches, embed_dim)

    return: encoded representation of the input sequence with same shape (batch_size, 16, 64)

decoder.py:

    Internal functionality:
    - defines a Transformer decoder block with:
        - masked multi-head self-attention over decoder inputs
        - cross-attention over encoder outputs
        - feed-forward network
        - residual connections + LayerNorm
    - embeds decoder token input sequence [<sos>, d1, d2, d3, d4]
    - applies learned positional encodings (1, 5, 64)
    return: logits of shape (batch_size, 5, 13) over vocabulary: [0-9, <sos>, <eos>, <empty>]

model_transformer.py

    Internal functionality:
      - central entry point combining encoder and decoder
      - embeds raw patch inputs (B, 16, 196) via nn.Linear(196, 64)
      - adds learnable positional encodings to embedded patches

      - feeds embedded image patches to encoder
      - feeds embedded decoder tokens to decoder

      return: sequence predictions of shape (batch_size, 5, 13)

s02_training.py

    Internal functionality:

evaluate.py:

    Internal functionality:

s03_inference.py:

    Internal functionality:
