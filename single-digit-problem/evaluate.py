import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from encoder import TransformerEncoder
from s01_mnist_dataset import MNISTDataset
from model import MNISTClassifier

#
#
# CONSTANTS
#
#

BATCH_SIZE = 128
EMBED_DIM = 32
HIDDEN_DIM = 24
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#
# EVALUATION FUNCTION
#
#

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Evaluation Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

#
#
# MAIN
#
#

if __name__ == "__main__":
    # load test data
    test_dataset = MNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # load model
    encoder = TransformerEncoder(NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, NUM_HEADS).to(device)
    model = MNISTClassifier(encoder).to(device)
    model.load_state_dict(torch.load("mnist_classifier.pth"))
    print("Loaded model from mnist_classifier.pth")

    # evaluate
    evaluate(model, test_loader)
