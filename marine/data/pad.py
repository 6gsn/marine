import torch
from torch.nn.utils.rnn import pad_sequence


class Padsequence(object):
    def __init__(
        self,
        input_keys,
        input_length_key,
        output_keys,
        num_classes,
        is_inference=False,
        padding_idx=0,
    ):
        self.input_keys = input_keys
        self.input_length_key = input_length_key
        self.output_keys = output_keys
        self.num_classes = num_classes
        self.is_inference = is_inference
        self.padding_idx = padding_idx

    def pad_feature(self, inputs):
        padded_feature = {}

        for key in self.input_keys:
            feature = [
                torch.tensor(features[key], dtype=torch.int64) for features in inputs
            ]

            if key in self.input_length_key:
                padded_feature[f"{key}_length"] = torch.tensor(
                    [len(f) for f in feature], dtype=torch.int64
                )

            padded_x = pad_sequence(
                feature,
                batch_first=True,
                padding_value=self.padding_idx,
            )
            padded_feature[key] = padded_x

        return padded_feature

    def __call__(self, batch):
        # sort by length
        if not self.is_inference:
            batch = sorted(
                batch,
                key=lambda x: len(x["features"][self.input_length_key]),
                reverse=True,
            )

        inputs = [x["features"] for x in batch]
        padded_inputs = self.pad_feature(inputs)

        if not self.is_inference:
            padded_outputs = {
                key: {
                    "label": pad_sequence(
                        [
                            # Covnert 1-based label (for pad)
                            torch.tensor(x["labels"][key] + 1, dtype=torch.long)
                            for x in batch
                        ],
                        batch_first=True,
                        padding_value=self.padding_idx,
                    ),
                    "length": torch.tensor([len(x["labels"][key]) for x in batch]),
                }
                for key in self.output_keys
            }
            script_ids = [x["ids"] for x in batch]
        else:
            padded_outputs = None
            script_ids = None

        morph_boundary = [x["features"]["morph_boundary"] for x in batch]

        return padded_inputs, padded_outputs, morph_boundary, script_ids
