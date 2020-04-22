import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_dim,
        n_layers=2,
        pad_idx=None,
        hidden_dim=128,
        embed_dim=300,
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()
        num_directions = 1 if not bidirectional else 2
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim * n_layers * num_directions,
                                output_dim)

    def forward(self, x, x_len):
        x = self.embedding(x)
        # Pad each sentences for a batch,
        # the final x with shape (seq_len, batch_size, embed_size)
        x = pack_padded_sequence(x, x_len)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        # NOTE: take the last hidden state of encoder as in seq2seq architecture.
        hidden_states, (h_n, c_c) = self.lstm(x)
        h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous()
        # h_n:(batch_size, hidden_size * num_layers * num_directions)
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.linear(h_n)
        return loggits
