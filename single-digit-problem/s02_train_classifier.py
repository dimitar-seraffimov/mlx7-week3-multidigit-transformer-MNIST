import torch
import wandb
from tqdm import tqdm
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from encoder import TransformerEncoder
from s01_mnist_dataset import MNISTDataset
from evaluate import evaluate
from model import MNISTClassifier

#
#
# CONSTANTS
#
#

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.003

EMBED_DIM = 32
HIDDEN_DIM = 24
NUM_LAYERS = 6
NUM_HEADS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

#
#
# TRAINING LOOP
#
#

def train_classifier(model, dataloader, optimizer, criterion, epoch):
  model.train()
  total_loss = 0

  progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

  for batch in progress_bar:
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    logits = model(inputs)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    wandb.log({"loss": loss.item(), "epoch": epoch + 1})
    
    progress_bar.set_postfix({"loss": loss.item()})

  avg_loss = total_loss / len(dataloader)
  print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
  

#
#
# MAIN
#
#

if __name__ == "__main__":
  wandb.init(
    project="mlx7-week3-multidigit-transformer",
    name=f"single-digit-problem-{timestamp}",
    config={
      "epochs": EPOCHS,
      "batch_size": BATCH_SIZE,
      "lr": LR,
      "embed_dim": EMBED_DIM,
      "hidden_dim": HIDDEN_DIM,
      "num_heads": NUM_HEADS,
      "num_layers": NUM_LAYERS
    }
  )
  # load dataset
  train_dataset = MNISTDataset(split="train")
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

  test_dataset = MNISTDataset(split="test")
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  
  # initialise encoder
  encoder = TransformerEncoder(NUM_LAYERS, EMBED_DIM, HIDDEN_DIM, NUM_HEADS).to(device)
  model = MNISTClassifier(encoder).to(device)
  wandb.watch(model, log="all") # log all gradients and parameters

  # initialise loss function and optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(EPOCHS):
    train_classifier(model, train_loader, optimizer, criterion, epoch)
  
  torch.save(model.state_dict(), "mnist_classifier.pth")
  artifact = wandb.Artifact(f"mnist-model-{timestamp}", type="model")
  artifact.add_file("mnist_classifier.pth")
  wandb.log_artifact(artifact)

  print("Model saved to mnist_classifier.pth")
  
  evaluate(model, test_loader)
  
  wandb.finish()