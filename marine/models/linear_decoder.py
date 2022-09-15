from torch import cat, nn


class LinearDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        prev_task_embedding_label_list=None,
        prev_task_embedding_label_size=None,
        prev_task_embedding_size=None,
        prev_task_dropout=None,
        padding_idx=0,
    ):
        super().__init__()
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

        # NOTE: output_size must includes size for [PAD]
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, logits, mask, prev_decoder_outputs=None, decoder_targets=None):
        if self.prev_task_embedding is not None:
            prev_decoder_output_embs = []

            for key in self.prev_task_embedding.keys():
                prev_decoder_output = prev_decoder_outputs[key]
                prev_decoder_output_emb = self.prev_task_embedding[key](
                    prev_decoder_output
                )

                if self.prev_task_dropout:
                    prev_decoder_output_emb = self.prev_task_dropout[key](
                        prev_decoder_output_emb
                    )

                prev_decoder_output_embs.append(prev_decoder_output_emb)

            logits = cat([logits] + prev_decoder_output_embs, dim=2)

        # Linear -> B * T * Output-size
        linear_logits = self.linear(logits)

        return linear_logits
