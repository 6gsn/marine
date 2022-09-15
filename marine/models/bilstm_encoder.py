from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, shared_with=None):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, embeddings, lengths):
        # LSTM -> B * T * Hidden-size
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=True
        )
        logits, _ = self.lstm(packed)
        logits, _ = pad_packed_sequence(logits, batch_first=True)

        return logits
