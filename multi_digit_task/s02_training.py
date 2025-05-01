import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_transformer import MultiDigitTransformer
from s01_multi_digit_dataset import MultiDigitDataset
import wandb
from tqdm import tqdm
import datetime

#
#
# SETUP
#
#

BATCH_SIZE = 32
EPOCHS = 5
LR = 0.0003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

#
#
# TRAINING LOOP
#
#

def train_epoch(model, dataloader, optimizer, criterion, epoch):
  model.train()
  total_loss = 0

  progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

  for patches, label in progress_bar:
    patch_input = patches.to(device)
    label_input = label[:, :-1].to(device)  # input: <sos> + 4 digits
    label_target = label[:, 1:].to(device)  # target: 4 digits + <eos>

    logits = model(patch_input, label_input)  # (B, 5, 13)

    loss = criterion(logits.reshape(-1, logits.size(-1)), label_target.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    wandb.log({"loss": loss.item(), "epoch": epoch + 1})
    progress_bar.set_postfix({"loss": loss.item()})

  avg_loss = total_loss / len(dataloader)
  print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

#
# MAIN
#

if __name__ == "__main__":
  wandb.init(
      project="mlx7-week3-multidigit-transformer",
      name=f"multi-digit-transformer-{timestamp}",
      config={
          "epochs": EPOCHS,
          "batch_size": BATCH_SIZE,
          "lr": LR
      }
  )

  train_dataset = MultiDigitDataset(split="train")
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

  model = MultiDigitTransformer().to(device)
  wandb.watch(model, log="all")

  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  criterion = nn.CrossEntropyLoss(ignore_index=-100)

  for epoch in range(EPOCHS):
    train_epoch(model, train_loader, optimizer, criterion, epoch)

  torch.save(model.state_dict(), "multidigit_transformer.pth")
  artifact = wandb.Artifact(f"multidigit-transformer-{timestamp}", type="model")
  artifact.add_file("multidigit_transformer.pth")
  wandb.log_artifact(artifact)

  print("Model saved to multidigit_transformer.pth")

  wandb.finish()