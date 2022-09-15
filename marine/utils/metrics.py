import torch
from marine.utils.util import convert_ap_based_accent_to_mora_based_accent
from torchmetrics import F1Score, Metric, MetricCollection


class SentenceLevelAccuracy(Metric):
    """Metrics to calculate sentence level accuray."""

    full_state_update: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor):
        """Update variables for accuracy."""

        # preds: (B, T), target: (B, T), mask: (B, T)
        # sequence_level_matchs: (B)
        sequence_level_matchs = torch.LongTensor(
            [
                (target[mask] == pred[mask]).all()
                for pred, target, mask in zip(preds, targets, masks)
            ]
        )

        self.correct += torch.sum(sequence_level_matchs)
        self.total += sequence_level_matchs.numel()  # == batch size

    def compute(self):
        """Compute accuracy using variables."""
        return self.correct.float() / self.total


class MultiTaskMetrics(object):
    """Metrics to calculate scores."""

    def __init__(
        self,
        phase,
        task_label_sizes,
        average="macro",
        accent_represent_mode="binary",
        require_ap_level_f1_score=False,
        device="cpu",
    ):
        self.phase = phase
        self.tasks = task_label_sizes.keys()
        self.accent_represent_mode = accent_represent_mode
        self.require_ap_level_f1_score = require_ap_level_f1_score
        self.device = device

        self.metrics = {
            task_name: MetricCollection(
                {
                    "mora_level_f1_score": F1Score(
                        num_classes=2,  # the AN label represents High/Low or non-AN/AN
                        average=average,
                    ).to(device),
                    "ap_level_f1_score": F1Score(
                        num_classes=task_label_size, average=average
                    ).to(device),
                    "sentence_level_accuracy": SentenceLevelAccuracy().to(device),
                }
                if task_name == "accent_status" and require_ap_level_f1_score
                else {
                    "mora_level_f1_score": F1Score(
                        num_classes=task_label_size, average=average
                    ).to(device),
                    "sentence_level_accuracy": SentenceLevelAccuracy().to(device),
                }
            ).to(device)
            for task_name, task_label_size in task_label_sizes.items()
        }

    def update(self, task, preds, targets, masks, padding_idx=0, **kwargs):
        """Update variables for accuracy."""
        assert task in self.tasks, f"Not initialized task: {task} not in {self.tasks}"

        masked_preds, masked_targets = preds[masks], targets[masks]

        # Verify that masked target label sequence not includes [PAD] token
        # TODO: This assert could be omited using `ignore_index` of `F1Score`
        # However, the option dosen't behave as explained until 2022/08/10
        # see https://github.com/Lightning-AI/metrics/issues/613
        assert padding_idx not in masked_targets

        if task == "accent_status" and self.require_ap_level_f1_score:
            mora_pred, mora_target = self._convert_ap_seq_to_mora_seq(
                preds, targets, masks, **kwargs
            )
            self.metrics[task]["mora_level_f1_score"].update(mora_pred, mora_target)
            self.metrics[task]["ap_level_f1_score"].update(masked_preds, masked_targets)
        else:
            self.metrics[task]["mora_level_f1_score"].update(
                masked_preds, masked_targets
            )

        self.metrics[task]["sentence_level_accuracy"].update(preds, targets, masks)

    def compute(self):
        """Compute accuracy using variables."""
        return {
            task_name: {
                score_name: metrics.compute().cpu().item()
                for score_name, metrics in self.metrics[task_name].items()
            }
            for task_name in self.tasks
        }

    def reset(self):
        """Reset all variables in metrics."""
        for task_name in self.tasks:
            for metrics in self.metrics[task_name].values():
                metrics.reset()

    def _convert_ap_seq_to_mora_seq(
        self,
        preds,
        targets,
        ap_seq_masks,
        predicted_accent_phrase_boundaries,
        target_accent_phrase_boundaries,
        mora_seq_masks,
    ):
        """Convert accent phrase-based accent status sequence to mora-based."""
        mora_preds = convert_ap_based_accent_to_mora_based_accent(
            preds,
            predicted_accent_phrase_boundaries,
            ap_seq_masks,
            mora_seq_masks,
            self.accent_represent_mode,
        )
        mora_targets = convert_ap_based_accent_to_mora_based_accent(
            targets,
            target_accent_phrase_boundaries,
            ap_seq_masks,
            mora_seq_masks,
            self.accent_represent_mode,
        )

        return (
            torch.LongTensor(mora_preds).to(self.device),
            torch.LongTensor(mora_targets).to(self.device),
        )
