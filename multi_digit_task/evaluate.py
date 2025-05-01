import torch
from torch.utils.data import DataLoader
from s01_multi_digit_dataset import MultiDigitDataset
from model_transformer import MultiDigitTransformer
from tqdm import tqdm
import wandb

#
#
# CONSTANTS
#
# 

VOCAB_SIZE = 13

index_to_label = {
  0: 0,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
  7: 7,
  8: 8,
  9: 9,
  10: "<sos>",
  11: "<eos>",
  12: "<empty>"
}

#
#
# EVALUATE
#
#

def evaluate(model, dataloader, device):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
      for patches, label in tqdm(dataloader, desc="Evaluating"):
          patch_input = patches.to(device)
          decoder_input = label[:, :-1].to(device)  # input tokens
          target = label[:, 1:].to(device) # expected output

          logits = model(patch_input, decoder_input)
          preds = torch.argmax(logits, dim=-1)

          mask = target != -100  # ignore padding if used
          correct += (preds == target).masked_select(mask).sum().item()
          total += mask.sum().item()

  acc = correct / total if total > 0 else 0.0
  print(f"Accuracy: {acc:.4f}")
  wandb.log({"eval_accuracy": acc})
  return acc

if __name__ == "__main__":
  from model_transformer import MultiDigitTransformer

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MultiDigitTransformer().to(device)
  model.load_state_dict(torch.load("multidigit_transformer.pth", map_location=device))

  test_dataset = MultiDigitDataset(split="test")
  test_loader = DataLoader(test_dataset, batch_size=32)

  evaluate(model, test_loader, device)
