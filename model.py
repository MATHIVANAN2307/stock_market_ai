import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = self.attn(x)                    # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)   # attention over time
        context = torch.sum(weights * x, dim=1)  # (batch, hidden_dim)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(128)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        return self.fc(out).float()
