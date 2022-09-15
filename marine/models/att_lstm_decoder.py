import torch
from marine.modules.attention import BahdanauAttention, ZoneOutCell
from marine.utils.util import get_ap_length
from torch import nn


class AttentionBasedLSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size=512,
        output_size=20,
        hidden_size=512,
        num_layers=2,
        attention_hidden_size=128,
        decoder_embedding_size=256,
        zoneout=0.1,
        prev_task_dropout=0.5,
        decoder_prev_out_dropout=0.5,
        prev_task_embedding_label_list=None,
        prev_task_embedding_label_size=None,
        prev_task_embedding_size=None,
        padding_idx=0,
    ):
        super().__init__()
        # NOTE: output_size must includes size for [PAD]
        self.output_size = output_size

        if prev_task_embedding_label_size:
            embeddings = {}
            dropouts = {}
            for key in prev_task_embedding_label_list:
                embeddings[key] = nn.Embedding(
                    prev_task_embedding_label_size[key],
                    prev_task_embedding_size[key],
                    padding_idx=padding_idx,
                )
                input_size += prev_task_embedding_size[key]

                if prev_task_dropout:
                    dropouts[key] = nn.Dropout(prev_task_dropout)

            self.prev_task_embedding = nn.ModuleDict(embeddings)

            if len(dropouts) > 0:
                self.prev_task_dropout = nn.ModuleDict(dropouts)
            else:
                self.prev_task_dropout = None

        else:
            self.prev_task_embedding = None
            self.prev_task_dropout = None

        self.attention = BahdanauAttention(
            input_size,
            hidden_size,
            attention_hidden_size,
        )

        # in_dim: output-size + [PAD + SOS]
        self.decoder_embedding = nn.Embedding(
            self.output_size + 1, decoder_embedding_size, padding_idx=padding_idx
        )

        self.decoder_prev_out_dropout = nn.Dropout(decoder_prev_out_dropout)

        # Setup autogressive LSTM layer
        self.lstm = nn.ModuleList()
        for layer in range(num_layers):
            lstm = nn.LSTMCell(
                input_size + decoder_embedding_size if layer == 0 else hidden_size,
                hidden_size,
            )
            self.lstm += [ZoneOutCell(lstm, zoneout)]

        # Setup feature projection layer
        project_size = input_size + hidden_size
        # out_dim: output-size + [PAD]
        self.projection = nn.Linear(project_size, self.output_size, bias=False)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, encoder_outputs, mask, prev_task_outputs, decoder_targets=None):
        is_inference = decoder_targets is None
        ap_lengths = get_ap_length(prev_task_outputs["accent_phrase_boundary"], mask)

        if is_inference:
            max_decoder_time_steps = max(ap_lengths)
        else:
            max_decoder_time_steps = decoder_targets.shape[1]

        if self.prev_task_embedding is not None:
            prev_task_output_embs = []
            for key in self.prev_task_embedding.keys():
                prev_task_output = prev_task_outputs[key]
                prev_task_output_emb = self.prev_task_embedding[key](prev_task_output)

                if self.prev_task_dropout:
                    prev_task_output_emb = self.prev_task_dropout[key](
                        prev_task_output_emb
                    )

                prev_task_output_embs.append(prev_task_output_emb)

            encoder_outputs = torch.cat(
                [encoder_outputs] + prev_task_output_embs, dim=2
            )

        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outputs))
            c_list.append(self._zero_state(encoder_outputs))

        go_frame = encoder_outputs.new_zeros(encoder_outputs.size(0), dtype=torch.long)
        go_frame[:] = self.output_size  # SOS
        prev_out = go_frame

        self.attention.reset()

        outs, att_ws = [], []
        t = 0

        while True:
            att_c, att_w = self.attention(encoder_outputs, h_list[0], mask)

            decoder_emb = self.decoder_embedding(prev_out)
            decoder_emb = self.decoder_prev_out_dropout(decoder_emb)

            # LSTM
            xs = torch.cat([att_c, decoder_emb], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            hcs = torch.cat([h_list[-1], att_c], dim=1)

            outs.append(
                # (B, out_dim) -> (B, 1, out_dim)
                self.projection(hcs).view(
                    encoder_outputs.size(0),
                    -1,
                    self.output_size,
                )
            )
            att_ws.append(att_w)

            if is_inference:
                # List[(B, 1, out_dim)] -> (B, out_dim) -> (B, 1)
                prev_out = torch.argmax(outs[-1][:, -1, :], dim=1)
            else:
                # Teacher forcing
                # (B, Lmax) -> (B, 1)
                prev_out = decoder_targets[:, t]

            t += 1
            if t >= max_decoder_time_steps:
                break

        outs = torch.cat(outs, dim=1)  # (B, Lmax, out_dim)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        return outs, att_ws, ap_lengths
