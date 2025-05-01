import torch
from model_transformer import MultiDigitTransformer
from s01_multi_digit_dataset import label_to_index, index_to_label
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn.functional as F
import matplotlib.pyplot as plt


#
#
# SETUP
#
#

EXAMPLES = 8
EMBED_DIM = 64
NUM_PATCHES = 16
MAX_SEQ_LEN = 5
VOCAB_SIZE = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#
#
# INFERENCE
#
#

class DigitPredictor:
  def __init__(self, model_path="multidigit_transformer.pth"):
    self.model = MultiDigitTransformer().to(DEVICE)
    self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    self.model.eval()

  def predict(self, patch_tensor):
    """
    patch_tensor: (1, 16, 196)
    returns: list of predicted tokens (e.g. [7, 2, 1, <empty>])
    """
    with torch.no_grad():
        patch_tensor = patch_tensor.to(DEVICE)
        decoder_input = torch.tensor([[label_to_index["<sos>"]]], dtype=torch.long).to(DEVICE)

        output_tokens = []

        for _ in range(MAX_SEQ_LEN):
            logits = self.model(patch_tensor, decoder_input)  # (1, t, vocab)
            next_token_logits = logits[:, -1, :]  # (1, vocab)
            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1)
            output_tokens.append(next_token.item())

            if index_to_label[next_token.item()] == "<eos>":
                break

            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

    return [index_to_label[tok] for tok in output_tokens]

if __name__ == "__main__":
  from s01_multi_digit_dataset import MultiDigitDataset
  import random

  dataset = MultiDigitDataset(split="test")
  predictor = DigitPredictor()

  fig, axes = plt.subplots(2, EXAMPLES//2, figsize=(15, 6))
  axes = axes.flatten()

  for i in range(EXAMPLES):
      patches, label_tensor, composed_img = dataset[random.randint(0, len(dataset)-1)]
      input_tensor = patches.unsqueeze(0)  # (1, 16, 196)
      prediction = predictor.predict(input_tensor)
      label = [index_to_label[idx.item()] for idx in label_tensor[1:-1]]  # skip <sos> and <eos>

      axes[i].imshow(composed_img, cmap="gray")
      axes[i].axis("off")

      # add predictions and labels below the image
      axes[i].annotate(f"Pred: {''.join(map(str, prediction))}",
                       (0.5, -0.15), xycoords='axes fraction',
                       ha='center', fontsize=10)

      axes[i].annotate(f"GT: {''.join(map(str, label))}",
                       (0.5, -0.30), xycoords='axes fraction',
                       ha='center', fontsize=10)

      print(f"[{i+1}] Prediction: {prediction} | Ground Truth: {label}")

  plt.tight_layout()
  plt.show()