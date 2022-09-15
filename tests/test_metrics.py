import json
from logging import getLogger
from pathlib import Path
from typing import Dict

import pytest
import torch
from marine.utils.metrics import MultiTaskMetrics, SentenceLevelAccuracy
from numpy.testing import assert_almost_equal
from pkg_resources import resource_filename

logger = getLogger("test")
BASE_DIR = Path(resource_filename("marine", ""))


@pytest.fixture
def sentence_level_accuracy() -> SentenceLevelAccuracy:
    return SentenceLevelAccuracy()


@pytest.fixture
def high_low_multi_task_metrics() -> MultiTaskMetrics:
    phase = "train"
    task_label_sizes = {"accent_phrase_boundary": 3}

    return MultiTaskMetrics(phase, task_label_sizes, device="cpu")


@pytest.fixture
def ap_multi_task_metrics() -> MultiTaskMetrics:
    phase = "train"
    task_label_sizes = {"accent_status": 21}
    require_ap_level_f1_score = True

    return MultiTaskMetrics(
        phase,
        task_label_sizes,
        require_ap_level_f1_score=require_ap_level_f1_score,
        device="cpu",
    )


@pytest.fixture
def full_multi_task_metrics() -> MultiTaskMetrics:
    phase = "train"
    task_label_sizes = {
        "intonation_phrase_boundary": 3,
        "accent_phrase_boundary": 3,
        "accent_status": 21,
    }
    require_ap_level_f1_score = True
    device = "cpu"

    return MultiTaskMetrics(
        phase,
        task_label_sizes,
        require_ap_level_f1_score=require_ap_level_f1_score,
        device=device,
    )


@pytest.fixture
def test_log_sample() -> Dict:
    logs = None
    sample_path = BASE_DIR.parent / "tests" / "samples" / "test_log_sample.json"
    with open(sample_path, "r") as file:
        logs = json.load(file)
    return logs


def test_sentence_level_accuracy(sentence_level_accuracy):
    for pred, target, mask, expect in [
        (
            torch.tensor([[1, 2, 1, 1, 2, 1, 0], [1, 2, 1, 1, 1, 0, 0]]),
            torch.tensor([[1, 1, 2, 1, 1, 2, 0], [1, 1, 2, 1, 1, 0, 0]]),
            torch.tensor(
                [
                    [True, True, True, True, True, True, False],
                    [True, True, True, True, True, False, False],
                ],
            ),
            0.0,
        ),
        (
            torch.tensor([[1, 1, 2, 1, 1, 2, 0], [1, 2, 1, 1, 1, 0, 0]]),
            torch.tensor([[1, 1, 2, 1, 1, 2, 0], [1, 1, 2, 1, 1, 0, 0]]),
            torch.tensor(
                [
                    [True, True, True, True, True, True, False],
                    [True, True, True, True, True, False, False],
                ],
            ),
            0.5,
        ),
        (
            torch.tensor([[1, 1, 2, 1, 1, 2, -1], [1, 2, 1, 1, 1, 0, 0]]),
            torch.tensor([[1, 1, 2, 1, 1, 2, -1], [1, 2, 1, 1, 1, 0, 0]]),
            torch.tensor(
                [
                    [True, True, True, True, True, True, False],
                    [True, True, True, True, True, False, False],
                ],
            ),
            1.0,
        ),
    ]:
        sentence_level_accuracy.update(pred, target, mask)
        score = sentence_level_accuracy.compute()
        assert_almost_equal(score, expect)
        sentence_level_accuracy.reset()


def test_ap_muti_task_metrics(ap_multi_task_metrics):
    for task, ap_pred, ap_target, ap_mask, kwargs, expect in [
        (
            "accent_status",
            # NOTE: AP-based accent label requires remove padding index (=0)
            # e,g, torch.tensor([4]) means 3rd mora has accent nucleus
            torch.tensor([[1, 1, 2], [1, 3, 0]]),
            torch.tensor([[1, 1, 3], [1, 2, 0]]),
            torch.tensor([[True, True, True], [True, True, False]]),
            {
                # NOTE: accent phrase boundary label requires remove padding index (=0)
                # e,g, torch.tensor([1, 2, 1]) means 2nd mora has accent phrase boundary
                "predicted_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "target_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "mora_seq_masks": torch.tensor(
                    [
                        [True, True, True, True, True, True, True, False],
                        [True, True, True, True, True, False, False, False],
                    ]
                ),
            },
            {
                "ap_level_f1_score": 0.3333333432674408,
                "mora_level_f1_score": 0.40000003576278687,
                "sentence_level_accuracy": 0.0,
            },
        ),
        (
            "accent_status",
            torch.tensor([[1, 1, 3], [1, 2, -1]]),
            torch.tensor([[1, 1, 3], [1, 3, -1]]),
            torch.tensor([[True, True, True], [True, True, False]]),
            {
                "predicted_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "target_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "mora_seq_masks": torch.tensor(
                    [
                        [True, True, True, True, True, True, True, False],
                        [True, True, True, True, True, False, False, False],
                    ]
                ),
            },
            {
                "ap_level_f1_score": 0.5555555820465088,
                "mora_level_f1_score": 0.699999988079071,
                "sentence_level_accuracy": 0.5,
            },
        ),
        (
            "accent_status",
            torch.tensor([[1, 1, 3], [1, 3, 0]]),
            torch.tensor([[1, 1, 3], [1, 3, 0]]),
            torch.tensor([[True, True, True], [True, True, False]]),
            {
                "predicted_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "target_accent_phrase_boundaries": torch.tensor(
                    [[1, 2, 1, 1, 2, 1, 1, 0], [1, 2, 1, 1, 1, 0, 0, 0]]
                ),
                "mora_seq_masks": torch.tensor(
                    [
                        [True, True, True, True, True, True, True, False],
                        [True, True, True, True, True, False, False, False],
                    ]
                ),
            },
            {
                "ap_level_f1_score": 1.0,
                "mora_level_f1_score": 1.0,
                "sentence_level_accuracy": 1.0,
            },
        ),
    ]:
        ap_multi_task_metrics.update(task, ap_pred, ap_target, ap_mask, **kwargs)
        scores = ap_multi_task_metrics.compute()

        for score_name, expect_score in expect.items():
            assert_almost_equal(scores[task][score_name], expect_score)

        ap_multi_task_metrics.reset()


def test_ap_muti_task_metrics_by_log(test_log_sample, full_multi_task_metrics):
    """Unit test for MultiTaskMetrics by sample"""

    def _extract_label(key, log):
        label = [int(v) for v in log[key].split(",")]
        # (T) -> (1, T)
        label = torch.tensor(label).unsqueeze(0).to("cpu")
        return label

    expected_scores = {
        "intonation_phrase_boundary": {
            "mora_level_f1_score": 0.9472237825393677,
            "sentence_level_accuracy": 0.7064254283905029,
        },
        "accent_phrase_boundary": {
            "mora_level_f1_score": 0.993394672870636,
            "sentence_level_accuracy": 0.875554,
        },
        "accent_status": {
            "ap_level_f1_score": 0.6181702017784119,
            "mora_level_f1_score": 0.9536119699478149,
            "sentence_level_accuracy": 0.6894387006759644,
        },
    }

    for script_status in test_log_sample.values():
        for task_name, task_status in script_status.items():
            preds = _extract_label("predict", task_status)
            targets = _extract_label("target", task_status)

            if task_name == "accent_status":
                predicted_ap_boundary = _extract_label(
                    "predict", script_status["accent_phrase_boundary"]
                )
                target_ap_bondary = _extract_label(
                    "target", script_status["accent_phrase_boundary"]
                )
                # ap_mask: (1, T_ap)
                ap_seq_mask = torch.full(targets.shape, True).to("cpu")
                # mora_mask: (1, T_mora)
                mora_seq_mask = torch.full(target_ap_bondary.shape, True).to("cpu")

                full_multi_task_metrics.update(
                    task_name,
                    preds,
                    targets,
                    ap_seq_mask,
                    predicted_accent_phrase_boundaries=predicted_ap_boundary,
                    target_accent_phrase_boundaries=target_ap_bondary,
                    mora_seq_masks=mora_seq_mask,
                )
            else:
                masks = torch.full(targets.shape, True).to("cpu")
                full_multi_task_metrics.update(task_name, preds, targets, masks)

    scores = full_multi_task_metrics.compute()

    # verify keys the metric has is correct
    assert expected_scores.keys() == scores.keys()

    for task_name, task_status in expected_scores.items():
        for score_name, score in task_status.items():
            assert_almost_equal(score, scores[task_name][score_name])
