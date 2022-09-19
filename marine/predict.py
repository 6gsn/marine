import re
import warnings
from pathlib import Path

import torch
from marine.data.feature.feature_set import FeatureSet
from marine.data.pad import Padsequence
from marine.models import (
    AttentionBasedLSTMDecoder,
    CRFDecoder,
    LinearDecoder,
    init_model,
)
from marine.utils.openjtalk_util import convert_open_jtalk_format_label
from marine.utils.post_process import apply_postprocess_dict, load_postprocess_vocab
from marine.utils.pretrained import retrieve_pretrained_model
from marine.utils.util import (
    _convert_ap_based_accent_to_mora_based_accent,
    convert_label_by_accent_representation_model,
    expand_word_label_to_mora,
    sequence_mask,
)
from omegaconf import OmegaConf
from pkg_resources import resource_filename
from torch.nn.utils.rnn import pad_sequence

BASE_DIR = Path(resource_filename("marine", ""))
DEFAULT_POSTPROCESS_VOCAB_DIR = BASE_DIR / "dict"


class Predictor:
    """Interface for inference of accent model."""

    def __init__(
        self,
        model_dir=None,
        version=None,
        postprocess_vocab_dir=None,
        device="cpu",
        skip_post_process=False,
    ):
        self.setup_model(model_dir, version, device)
        self.setup_postprocess_vocab(postprocess_vocab_dir, skip_post_process)

    def setup_model(self, model_dir, version, device):
        if model_dir is None:
            if version is None:
                self.model_dir = Path(retrieve_pretrained_model())
            else:
                self.model_dir = Path(retrieve_pretrained_model(version))
        elif isinstance(model_dir, str):
            self.model_dir = Path(model_dir)

        assert (
            isinstance(self.model_dir, Path) and self.model_dir.exists()
        ), f"Model directory doesn't exists: {self.model_dir.as_posix()}"

        self.device = device
        self.config = OmegaConf.load(self.model_dir / "config.yaml")
        self.tasks = self.config.data.output_keys

        if self.config.model.vocab_path is None:
            self.config.model.vocab_path = str(self.model_dir / "vocab.pkl")

        self.feature_set = FeatureSet(
            self.config.model.vocab_path,
            feature_table_key=self.config.data.feature_table_key,
            feature_keys=self.config.data.input_keys,
        )
        self.model = init_model(self.tasks, self.config, self.feature_set, self.device)
        self._load_states()

        self.collate_fn = Padsequence(
            self.config.data.input_keys,
            self.config.data.input_length_key,
            self.config.data.output_keys,
            self.config.data.output_sizes,
            is_inference=True,
        )

    def _load_states(self):
        states = torch.load(self.model_dir / "model.pth", map_location=self.device)
        self.model.load_state_dict(states["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def setup_postprocess_vocab(self, postprocess_vocab_dir, skip_post_process):
        # Setup for vocab for post-process
        if postprocess_vocab_dir is None:
            self.postprocess_vocab_dir = DEFAULT_POSTPROCESS_VOCAB_DIR

        elif isinstance(postprocess_vocab_dir, str):
            self.postprocess_vocab_dir = Path(postprocess_vocab_dir)

        assert (
            self.postprocess_vocab_dir.exists()
        ), f"Vocab directory doesn't exists: {self.postprocess_vocab_dir.as_posix()}"

        if skip_post_process:
            self.postprocess_vocab, self.epostprocess_targets = None, None
        else:
            self.postprocess_vocab = load_postprocess_vocab(
                self.postprocess_vocab_dir, self.tasks
            )
            self.postprocess_targets = {
                key: re.compile(f"({'|'.join(vocab.keys())})") if vocab else None
                for key, vocab in self.postprocess_vocab.items()
            }

    @torch.no_grad()
    def predict(
        self,
        sentences,
        accent_represent_mode="binary",
        annotates=None,
        require_open_jtalk_format=False,
    ):
        if accent_represent_mode not in ["binary", "high_low"]:
            raise NotImplementedError(
                (
                    f"Not supported representation mode {accent_represent_mode}:"
                    " Representation mode must be selected in binary and high_low"
                )
            )

        if require_open_jtalk_format and accent_represent_mode != "binary":
            warnings.warn(
                (
                    "If you want the format for OpenJTalk,",
                    "`accent_represent_mode` will be fixed as `binary`",
                )
            )
            accent_represent_mode = "binary"

        result = {}

        inputs, morph_boundary = self.extract_feature(sentences)
        result["mora"] = self.convert_to_mora(inputs)

        if annotates is not None:
            annotates = self.pad_annotate_label(
                annotates, mora=result["mora"], morph_boundary=morph_boundary
            )

        for task in self.tasks:
            if annotates is not None and task in annotates.keys():
                real_labels = self.convert_to_label(
                    task,
                    annotates[task],
                    inputs["mask"],
                    prev_task_outputs=inputs["prev_decoder_outputs"],
                    accent_represent_mode=accent_represent_mode,
                )
                prev_output = annotates[task]

            else:
                outputs = self.model(task, **inputs)

                if isinstance(self.model.decoders[task], CRFDecoder):
                    _, logits = outputs
                    ap_lengths, ap_outputs = None, None

                elif isinstance(self.model.decoders[task], LinearDecoder):
                    logits = outputs
                    ap_lengths, ap_outputs = None, None

                elif isinstance(self.model.decoders[task], AttentionBasedLSTMDecoder):
                    logits, _, ap_lengths = outputs
                    ap_outputs = inputs["prev_decoder_outputs"][
                        "accent_phrase_boundary"
                    ]

                real_labels = self.convert_to_label(
                    task,
                    logits,
                    inputs["mask"],
                    moras=result["mora"],
                    ap_lengths=ap_lengths,
                    ap_outputs=ap_outputs,
                    prev_task_outputs=inputs["prev_decoder_outputs"],
                    accent_represent_mode=accent_represent_mode,
                )

                if self.postprocess_vocab:
                    target = self.postprocess_targets[task]
                    vocab = self.postprocess_vocab[task]

                    if vocab:
                        for index, (nodes, label, mora, boundary) in enumerate(
                            zip(sentences, real_labels, result["mora"], morph_boundary)
                        ):
                            real_labels[index] = apply_postprocess_dict(
                                task,
                                nodes,
                                label,
                                mora,
                                boundary,
                                target,
                                vocab,
                                accent_represent_mode,
                            )

                prev_output = pad_sequence(
                    [torch.tensor(label) for label in real_labels], batch_first=True
                ).to(self.device)

            result[task] = real_labels
            inputs["prev_decoder_outputs"][task] = prev_output

        if require_open_jtalk_format:
            result = convert_open_jtalk_format_label(result, morph_boundary)

        return result

    def extract_feature(self, sentences):
        batch = [
            {
                "features": self.feature_set.convert_nodes_to_feature(nodes),
                "labels": None,
            }
            for nodes in sentences
        ]
        inputs, _, morph_boundary, _ = self.collate_fn(batch)

        embeddings = {
            key: inputs[key].to(self.device) for key in self.config.data.input_keys
        }

        inputs = {
            "embedding_features": embeddings,
            "lengths": inputs["mora_length"].cpu(),
            "mask": sequence_mask(inputs["mora_length"]).to(self.device),
            "prev_decoder_outputs": {},
        }

        return inputs, morph_boundary

    def convert_to_mora(self, inputs):
        return [
            list(self.feature_set.convert_id_to_feature("mora", mora[mask].tolist()))
            for mora, mask in zip(inputs["embedding_features"]["mora"], inputs["mask"])
        ]

    def convert_to_label(
        self,
        task,
        outputs,
        mora_masks,
        moras=None,
        ap_lengths=None,
        ap_outputs=None,
        prev_task_outputs=None,
        accent_represent_mode="binary",
    ):
        predicts = []

        for index, (mora_mask, padded_predict) in enumerate(zip(mora_masks, outputs)):
            # for annotated label
            if len(padded_predict.shape) != 2:
                predict = padded_predict[mora_mask]
            else:
                if task == "accent_status":
                    # when the decoder for AN estimates AP-based label,
                    # convert AP-based label to mora-mased label
                    if ap_lengths is not None:
                        predict = padded_predict[: ap_lengths[index]]
                        ap_output = ap_outputs[index][mora_mask]
                        predict = torch.argmax(predict, dim=1)
                        predict = _convert_ap_based_accent_to_mora_based_accent(
                            predict,
                            ap_output,
                            mode=accent_represent_mode,
                            mora=moras[index],
                        )
                    # when the decoder for AN estimates mora-based label
                    # and `accent_represent_mode` is different
                    # between configuration for inference and model,
                    # convert the label to follow the inference setting
                    elif self.config.data.represent_mode != accent_represent_mode:
                        predict = padded_predict[mora_mask]
                        accent_phrase_boundary = prev_task_outputs[
                            "accent_phrase_boundary"
                        ][index][mora_mask]
                        predict = torch.argmax(predict, dim=1)
                        predict = convert_label_by_accent_representation_model(
                            predict,
                            accent_phrase_boundary,
                            moras[index],
                            current_accent_represent_mode=self.config.data.represent_mode,
                            target_accent_represent_mode=accent_represent_mode,
                        )
                    else:
                        predict = padded_predict[mora_mask]
                        predict = torch.argmax(predict, dim=1)
                        # convert 0-based label
                        predict = predict - 1
                else:
                    predict = padded_predict[mora_mask]
                    predict = torch.argmax(predict, dim=1)
                    # convert 0-based label
                    predict = predict - 1

            predicts.append((predict).tolist())

        return predicts

    def pad_annotate_label(self, annotates, mora, morph_boundary):
        for key in annotates.keys():
            token_type = annotates[key]["token_type"]
            annotate = annotates[key]["labels"]

            if token_type == "morph":
                annotate = expand_word_label_to_mora(
                    annotate, mora, morph_boundary, key
                )
            elif token_type != "mora":
                raise ValueError(f"Token type must be morph or mora: {token_type}")

            if isinstance(annotate[0], list):
                annotate = [torch.tensor(ann) for ann in annotate]
            elif not isinstance(annotates[key][0], torch.Tensor):
                raise ValueError(
                    (
                        "Annoate labels must be List[List] or List[tensor]:"
                        f" {type(annotates[key][0])}"
                    )
                )

            annotates[key] = pad_sequence(annotate, batch_first=True).to(self.device)

        return annotates
