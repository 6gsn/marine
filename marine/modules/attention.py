import torch
from torch import nn
from torch.nn import functional as F


class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h


class BahdanauAttention(nn.Module):
    """Bahdanau-style attention
    This is an attention mechanism originally used in Tacotron.
    Args:
        encoder_dim (int): dimension of encoder outputs
        decoder_dim (int): dimension of decoder outputs
        hidden_dim (int): dimension of hidden state
    """

    def __init__(self, encoder_dim=512, decoder_dim=1024, hidden_dim=128):
        super().__init__()
        self.mlp_enc = nn.Linear(encoder_dim, hidden_dim)
        self.mlp_dec = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def reset(self):
        """Reset the internal buffer"""
        self.processed_memory = None

    def forward(
        self,
        encoder_outs,
        decoder_state,
        mask=None,
    ):
        """Forward step
        Args:
            encoder_outs (torch.FloatTensor): encoder outputs
            src_lens (list): length of each input batch
            decoder_state (torch.FloatTensor): decoder hidden state
            mask (torch.FloatTensor): mask for padding
        """

        if self.processed_memory is None:
            self.processed_memory = self.mlp_enc(encoder_outs)

        decoder_state = self.mlp_dec(decoder_state).unsqueeze(1)

        erg = self.w(torch.tanh(self.processed_memory + decoder_state)).squeeze(-1)

        if mask is not None:
            # invert mask
            mask = ~mask
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights
