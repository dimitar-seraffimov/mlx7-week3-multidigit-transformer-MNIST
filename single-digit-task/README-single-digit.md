Step 01:

s01_mnist_dataset.py

    Internal functionality:

        - download MNIST parquet files from HuggingFace - https://huggingface.co/datasets/ylecun/mnist/tree/main/mnist
        - convert 28x28 images to 16 patches (7x7 each)
        - flatten each patch (1x49)
        - apply a nn.Linear(49, 32) transformation
        - add random positional encodings
        return: torch.Tensor of shape (16, 32) for each image.

Step 02:

s02_train_classifier.py

    Internal functionality:

        - load the MNIST data (already patch-embedded with positional encoding)
        - pass each batch through the encoder stack, number of encoder blocks = 6 (defined at NUM_LAYERS)
        - aggregate patch-level embeddings to form a single image-level representation
        - apply a classifier (nn.Linear(EMBED_DIM, 10)) to predict the digit (0–9)
        - train with standard cross-entropy loss

s03_inference.py:

    Internal functionality:

        - load the saved model
        - load a single image
        - preprocess it just like training
        - run it through the model
        return: the predicted digit

The encoder architecture setup:
encoder.py

    Internal functionality:
    - define a multi-head self-attention encoder block (with nn.Linear projections for Q, K, V)
    for each head:
        - compute attention scores using scaled dot-product (Q × Kᵀ / √d)
        - apply softmax to get attention weights
        - compute weighted sum of V values using the attention scores
    - concatenate outputs from all attention heads
    - apply a final linear projection back to embedding size (24 → 32)
    - add a feed-forward network (MLP) with GELU activation
    - use residual connections and layer normalization after both attention and FFN
    - stack multiple encoder blocks (in this example code -> 6 layers) using nn.ModuleList

    - accepts input of shape (batch_size, num_patches, embed_dim)
    return: encoded representation of the input sequence with same shape (B, 16, 32)

evaluate.py

    Internal functionality:
    - load the trained model
    - load the test dataset (MNIST test split)
    - run inference on the test set
    - compute accuracy and optionally log to Weights & Biases (wandb)
