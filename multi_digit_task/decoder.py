import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
# CONFIG
#
#

BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 24
NUM_LAYERS = 6
NUM_HEADS = 4
NUM_PATCHES = 16
VOCAB_SIZE = 13  # 0-9 and <sos>, <eos>, <empty>
MAX_SEQ_LEN = 5  # <sos> + 4 digits (<eos> is not passed to the decoder)

#
#
# DECODER BLOCK
#
#

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        # masked self-attention (Q, K, V from decoder input)
        self.self_q_proj = nn.Linear(embed_dim, hidden_dim)
        self.self_k_proj = nn.Linear(embed_dim, hidden_dim)
        self.self_v_proj = nn.Linear(embed_dim, hidden_dim)
        self.self_out_proj = nn.Linear(hidden_dim, embed_dim)

        # cross attention (Q from decoder input, K,V from encoder output)
        self.cross_q_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_k_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_v_proj = nn.Linear(embed_dim, hidden_dim)
        self.cross_out_proj = nn.Linear(hidden_dim, embed_dim)

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output):
        B, T, _ = x.size()

        # masked self-attention (Q, K, V from decoder input)
        q = self.self_q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        x = self.norm1(x + self.self_out_proj(attn_output))

        # cross attention (Q from decoder input, K,V from encoder output)
        q = self.cross_q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.cross_k_proj(encoder_output).view(B, encoder_output.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.cross_v_proj(encoder_output).view(B, encoder_output.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        cross_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        cross_weights = F.softmax(cross_scores, dim=-1)
        cross_output = torch.matmul(cross_weights, v).transpose(1, 2).contiguous().view(B, T, -1)
        x = self.norm2(x + self.cross_out_proj(cross_output))

        # feed-forward network
        x = self.norm3(x + self.ffn(x))
        return x

#
#
# TRANSFORMER DECODER
#
#

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, hidden_dim, num_heads, vocab_size, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, decoder_input, encoder_output):
        x = self.token_embedding(decoder_input) + self.position_embedding[:, :decoder_input.size(1), :]
        for layer in self.layers:
            x = layer(x, encoder_output)
        logits = self.output_linear(x)
        return logits

if __name__ == "__main__":
    decoder = TransformerDecoder(
        num_layers=NUM_LAYERS,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    decoder_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))
    encoder_output = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM)
    output = decoder(decoder_input, encoder_output)
    print("Decoder output shape:", output.shape) # (batch_size, max_seq_len, vocab_size) -> 32 x 5 x 13