import torch
import random
import matplotlib.pyplot as plt
from encoder import TransformerEncoder
from s01_mnist_dataset import MNISTDataset
from s02_train_classifier import MNISTClassifier

#
#
# CONSTANTS
#
#

EMBED_DIM = 32
HIDDEN_DIM = 24
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#
# INFERENCE FUNCTION
#
#

def run_inference():
    # load test dataset
    dataset = MNISTDataset(split="test")
    idx = random.randint(0, len(dataset) - 1)
    embedded, label = dataset[idx]  # (17, 32), int

    # we need the raw image for visualization
    img = dataset.images[idx]["bytes"]
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(img)).convert("L")

    # prepare input
    embedded = embedded.unsqueeze(0).to(DEVICE)  # (1, 17, 32)

    # load trained model
    encoder = TransformerEncoder(NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, NUM_HEADS)
    model = MNISTClassifier(encoder)
    model.load_state_dict(torch.load("mnist_classifier.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # run inference
    with torch.no_grad():
        logits = model(embedded)         # (1, 10)
        prediction = torch.argmax(logits, dim=1).item()

    # print result
    print(f"Predicted: {prediction}, Ground Truth: {label}")

    # show image
    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {prediction} | Label: {label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_inference()
