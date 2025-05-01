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
    - accept input of shape (batch_size, num_patches, embed_dim)

    return: encoded representation of the input sequence with same shape (batch_size, 16, 64)

decoder.py:

    Internal functionality:
    - define a Transformer decoder block with:
        - masked multi-head self-attention over decoder inputs
        - cross-attention over encoder outputs
        - feed-forward network
        - residual connections + LayerNorm
    - embed decoder token input sequence [<sos>, d1, d2, d3, d4]
    - apply learned positional encodings (1, 5, 64)
    return: logits of shape (batch_size, 5, 13) over vocabulary: [0-9, <sos>, <eos>, <empty>]

model_transformer.py

    Internal functionality:
      - central entry point combining encoder and decoder
      - embed raw patch inputs (B, 16, 196) via nn.Linear(196, 64)
      - add learnable positional encodings to embedded patches

      - feed embedded image patches to encoder
      - feed embedded decoder tokens to decoder

      return: sequence predictions of shape (batch_size, 5, 13)

s02_training.py

    Internal functionality:
      - load training dataset (MultiDigitDataset) and prepares DataLoader
      - initialize MultiDigitTransformer (which handles patch embedding and encoder/decoder)
      - passe raw image patch vectors (B, 16, 196) and decoder inputs [<sos>, digit1, ..., digit4]
      - compute cross-entropy loss between decoder output logits and ground-truth token sequence
      - log training loss and epoch via Weights & Biases (wandb)
      - save trained model and upload it to W&B as an artifact

evaluate.py:

    Internal functionality:
      - load trained model from .pth file
      - run evaluation over the test dataset (MultiDigitDataset)
      - compute token-level prediction accuracy (pred == target)
      - Log final accuracy to wandb
        CLI executable: python evaluate.py evaluate the latest model checkpoint

s03_inference.py:

    Internal functionality:
      - load a trained MultiDigitTransformer model
      - define a DigitPredictor class that performs greedy decoding using encoder outputs and a <sos>-initiated decoder loop
      - predict digit sequences from image patch tensors (1, 16, 196)
      - sample 8 random test images from the dataset and visualize:
        - input image (56×56)
        - print digit sequence below the image
        - ground truth label below for comparison
    EXAMPLE:
    ![Prediction results - 8 digits](multi_digit_task/progress_imgs/inference.png)
