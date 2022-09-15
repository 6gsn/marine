# coding: utf-8

from torch import nn


class BaseModel(nn.Module):
    def __init__(self, embedding, encoders, decoders):
        super().__init__()
        self.embedding = embedding
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

    def forward(
        self,
        task,
        embedding_features,
        lengths,
        mask,
        prev_decoder_outputs=None,
        decoder_targets=None,
    ):
        embeddings = self.embedding(**embedding_features)
        encoder_outputs = self.encoders[task](embeddings, lengths)
        decoder_outputs = self.decoders[task](
            encoder_outputs, mask, prev_decoder_outputs, decoder_targets
        )

        return decoder_outputs
