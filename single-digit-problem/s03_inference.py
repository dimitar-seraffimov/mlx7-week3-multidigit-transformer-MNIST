import torch
import random
import matplotlib.pyplot as plt
from encoder import TransformerEncoder
from s01_mnist_dataset import MNISTDataset
from model import MNISTClassifier
from PIL import Image
import io

#
#
# CONSTANTS
#
#

EMBED_DIM = 32
HIDDEN_DIM = 24
NUM_LAYERS = 6
NUM_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#
# INFERENCE FUNCTION
#
#

def run_inference(num_samples=10):
    # load test dataset
    dataset = MNISTDataset(split="test")

    # load trained model
    encoder = TransformerEncoder(NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, NUM_HEADS)
    model = MNISTClassifier(encoder)
    model.load_state_dict(torch.load("mnist_classifier.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # create figure for visualisation
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(num_samples):
      idx = random.randint(0, len(dataset) - 1)
      embedded = dataset[idx][0] # (17, 32)
      label = dataset.labels[idx] # int

      # get raw image
      raw_img = dataset.images[idx]["bytes"]
      img = Image.open(io.BytesIO(raw_img)).convert("L")

      # inference 
      embedded = embedded.unsqueeze(0).to(DEVICE) # (1, 17, 32)
      with torch.no_grad():
        logits = model(embedded) # (1, 10)
        prediction = torch.argmax(logits, dim=1).item()
      
      # plot image with prediction
      axes[i].imshow(img, cmap="gray")
      axes[i].set_title(f"Prediction: {prediction} | Ground Truth: {label}", fontsize=12)
      axes[i].axis("off")

      print(f"[{i+1}] Prediction: {prediction} | Ground Truth: {label}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()
