The files are listed in their order of creation:
s01_multi_digit_dataset.py

    Internal functionality:
      - download and load the MNIST dataset
      - create a black image to handle empty space (no-digits) scenario
      - randomly select and combine four digit images into a single 56×56 image
      - divide the combined image into 16 patches of size 14×14
      - flatten each patch into a 64-dimensional vector
      - return the flattened patches and the corresponding target sequence [<sos>, digit1, digit2, ..., <eos>]

encoder.py:

    Internal functionality:

decoder.py:

    Internal functionality:

model_transformer.py

    Internal functionality:
      - patches embedding
      - add positional encodings to each patch embedding

s02_training.py

    Internal functionality:

evaluate.py:

    Internal functionality:

s03_inference.py:

    Internal functionality: