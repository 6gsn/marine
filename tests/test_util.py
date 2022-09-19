import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from marine.data.feature.feature_set import FeatureSet
from marine.data.feature.feature_table import (
    is_adjective,
    is_noun,
    is_verb,
    parse_accent_con_type,
)
from marine.models.util import init_model
from marine.utils.openjtalk_util import (
    convert_njd_feature_to_marine_feature,
    convert_open_jtalk_format_label,
)
from marine.utils.util import (
    _calculate_multiple_task_scores,
    _convert_ap_based_accent_to_mora_based_accent,
    convert_label_by_accent_representation_model,
    expand_word_label_to_mora,
    get_accent_nucleus_in_binary_accent_stauts_seq,
    get_accent_nucleus_in_high_low_accent_stauts_seq,
)
from numpy.testing import assert_almost_equal
from omegaconf import DictConfig
from pkg_resources import resource_filename

BASE_DIR = Path(resource_filename("marine", ""))


@pytest.fixture
def default_config() -> DictConfig:
    config_path = BASE_DIR / "bin" / "conf" / "train" / "config.yaml"

    # initialize config
    with initialize_config_dir(config_dir=str(config_path.parent)):
        config = compose(config_name=config_path.name)
    GlobalHydra.instance().clear()

    return config


@pytest.fixture
def default_vocab_path() -> Path:
    return (
        BASE_DIR.parent
        / "recipe"
        / "common"
        / "database"
        / "20220912_jsut_vocab_min_2"
        / "vocab.pkl"
    )


@pytest.fixture
def test_log_sample() -> Dict:
    logs = None
    sample_path = BASE_DIR.parent / "tests" / "samples" / "test_log_sample.json"
    with open(sample_path, "r") as file:
        logs = json.load(file)
    return logs


def test_label_expanding():
    """Test function to convert mora-based IP sequence to mora-based."""
    for (labels, moras, word_boundaries), expected in [
        (
            # Case: the IP boundary annotated at middle
            (
                [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                [
                    [
                        "ワ",
                        "タ",
                        "シ",
                        "タ",
                        "チ",
                        "ガ",
                        "ム",
                        "ス",
                        "メ",
                        "ダ",
                        "ト",
                        "オ",
                        "モ",
                        "ッ",
                        "テ",
                        "イ",
                        "タ",
                    ]
                ],
                [
                    np.array(
                        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                        dtype=np.uint8,
                    ),
                ],
            ),
            [[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ),
        (
            # Case: IP boundary annotated at last
            (
                [[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]],
                [
                    [
                        "ワ",
                        "タ",
                        "シ",
                        "タ",
                        "チ",
                        "ガ",
                        "ム",
                        "ス",
                        "メ",
                        "ダ",
                        "ト",
                        "オ",
                        "モ",
                        "ッ",
                        "テ",
                        "イ",
                        "タ",
                    ],
                ],
                [
                    np.array(
                        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                        dtype=np.uint8,
                    ),
                ],
            ),
            [[1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ),
    ]:
        mora_labels = expand_word_label_to_mora(
            labels, moras, word_boundaries, "intonation_phrase_boundary"
        )
        assert expected == mora_labels


def test_accent_conversion():
    """Test function to convert ap-based accent sequence to mora-based."""
    for (
        ap_accents,
        accent_phrase_boundaries,
        accent_represent_mode,
        moras,
    ), expected in [
        # For binary
        # Correct label
        (
            (
                # NOTE: AP-based accent label requires remove padding index (=0)
                # e,g, torch.tensor([4]) means 3rd mora has accent nucleus
                torch.tensor([4]),
                # NOTE: accent phrase boundary label requires remove padding index (=0)
                # e,g, torch.tensor([1, 2, 1]) means 2nd mora has accent phrase boundary
                torch.tensor([1, 1, 1]),
                "binary",
                ["テ", "ス", "ト"],
            ),
            np.array([0, 0, 1]),
        ),
        # Wrong label: Out of mora
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1]),
                "binary",
                ["テ", "ス", "ト"],
            ),
            np.array([0, 0, 0]),
        ),
        # Wrong label: wrong locattion (long-vowel)
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1, 1]),
                "binary",
                ["テ", "ス", "ト", "ー"],
            ),
            np.array([0, 0, 0, 0]),
        ),
        # Wrong label: wrong locattion (sokuon)
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1, 1]),
                "binary",
                ["テ", "ス", "ト", "ッ"],
            ),
            np.array([0, 0, 0, 0]),
        ),
        # For high_low
        # Correct label
        (
            (
                torch.tensor([4]),
                torch.tensor([1, 1, 1]),
                "high_low",
                ["テ", "ス", "ト"],
            ),
            np.array([0, 1, 1]),
        ),
        # Wrong label: Out of mora
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1]),
                "high_low",
                ["テ", "ス", "ト"],
            ),
            np.array([0, 1, 1]),
        ),
        # Wrong label: wrong locattion (long-vowel)
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1, 1]),
                "high_low",
                ["テ", "ス", "ト", "ー"],
            ),
            np.array([0, 1, 1, 1]),
        ),
        # Wrong label: wrong locattion (sokuon)
        (
            (
                torch.tensor([5]),
                torch.tensor([1, 1, 1, 1]),
                "high_low",
                ["テ", "ス", "ト", "ッ"],
            ),
            np.array([0, 1, 1, 1]),
        ),
    ]:
        mora_accents = _convert_ap_based_accent_to_mora_based_accent(
            ap_accents,
            accent_phrase_boundaries,
            accent_represent_mode,
            moras,
        )

        assert np.all(mora_accents == expected)


def test_multiple_task_score_calculation(test_log_sample):
    tasks = ["intonation_phrase_boundary", "accent_phrase_boundary", "accent_status"]
    expected_scores = {
        "intonation_phrase_boundary+accent_phrase_boundary+accent_status": 0.5203101920236337,
        "intonation_phrase_boundary+accent_phrase_boundary": 0.6410635155096012,
        "accent_phrase_boundary+accent_status": 0.6790989660265879,
    }

    scores = _calculate_multiple_task_scores(tasks, test_log_sample)

    # verify multiple task key is correct
    assert scores.keys() == expected_scores.keys()

    # verify scoring is currect
    for task_key in expected_scores.keys():
        assert_almost_equal(
            scores[task_key]["sentence_level_accuracy"],
            expected_scores[task_key],
        )


def test_init_model(default_config, default_vocab_path):
    """Verify init_model() works with default config."""
    feature_set = FeatureSet(default_vocab_path)

    for task_group, is_train in [
        (
            ["intonation_phrase_boundary", "accent_phrase_boundary", "accent_status"],
            True,
        ),
        (
            ["accent_phrase_boundary", "accent_status"],
            True,
        ),
        (
            ["accent_status"],
            True,
        ),
        (
            ["intonation_phrase_boundary", "accent_phrase_boundary", "accent_status"],
            False,
        ),
        (
            ["accent_phrase_boundary", "accent_status"],
            False,
        ),
        (
            ["accent_status"],
            False,
        ),
    ]:
        output = init_model(
            task_group, default_config, feature_set, device="cpu", is_train=is_train
        )

        if is_train:
            model, _, _, _ = output
        else:
            model = output

        print(model)


def test_is_adjective():
    for pos_tag, expect in [
        ("助動詞:*:*:*", False),
        ("動詞:一般:*:*", False),
        ("名詞:助動詞語幹:*:*", False),
        ("形容詞:一般:*:*", True),
        ("形状詞:タリ:*:*", False),
    ]:
        assert is_adjective(pos_tag) == expect


def test_is_noun():
    for pos_tag, expect in [
        ("助動詞:*:*:*", False),
        ("動詞:一般:*:*", False),
        ("名詞:助動詞語幹:*:*", True),
        ("形容詞:一般:*:*", False),
        ("形状詞:タリ:*:*", False),
    ]:
        assert is_noun(pos_tag) == expect


def test_is_verb():
    for pos_tag, expect in [
        ("助動詞:*:*:*", False),
        ("動詞:一般:*:*", True),
        ("名詞:助動詞語幹:*:*", False),
        ("形容詞:一般:*:*", False),
        ("形状詞:タリ:*:*", False),
    ]:
        assert is_verb(pos_tag) == expect


def test_parse_accent_con_type():
    for a_con_type, pos_tag, expect in [
        ("動詞%F2@0,形容詞%F2@-1,名詞%F1", "助動詞:*:*:*", "[UNK]"),
        ("動詞%F2@0,形容詞%F2@-1,名詞%F1", "動詞:一般:*:*", "F2"),
        ("動詞%F2@0,形容詞%F2@-1,名詞%F1", "名詞:助動詞語幹:*:*", "F1"),
        ("動詞%F2@0,形容詞%F2@-1,名詞%F1", "形容詞:一般:*:*", "F2"),
        ("動詞%F2@0,形容詞%F2@-1,名詞%F1", "形状詞:タリ:*:*", "[UNK]"),
    ]:
        assert parse_accent_con_type(a_con_type, pos_tag) == expect


def test_convert_njd_feature_to_marine_feature():
    for features, expect in [
        (
            [
                {
                    "string": "これ",
                    "pos": "名詞",
                    "pos_group1": "代名詞",
                    "pos_group2": "一般",
                    "pos_group3": "*",
                    "ctype": "*",
                    "cform": "*",
                    "orig": "これ",
                    "read": "コレ",
                    "pron": "コレ",
                    "acc": 0,
                    "mora_size": 2,
                    "chain_rule": "C3",
                    "chain_flag": -1,
                },
                {
                    "string": "は",
                    "pos": "助詞",
                    "pos_group1": "係助詞",
                    "pos_group2": "*",
                    "pos_group3": "*",
                    "ctype": "*",
                    "cform": "*",
                    "orig": "は",
                    "read": "ハ",
                    "pron": "ワ",
                    "acc": 0,
                    "mora_size": 1,
                    "chain_rule": "名詞%F1/動詞%F2@0/形容詞%F2@0",
                    "chain_flag": 1,
                },
                {
                    "string": "テスト",
                    "pos": "名詞",
                    "pos_group1": "サ変接続",
                    "pos_group2": "*",
                    "pos_group3": "*",
                    "ctype": "*",
                    "cform": "*",
                    "orig": "テスト",
                    "read": "テスト",
                    "pron": "テス’ト",
                    "acc": 1,
                    "mora_size": 3,
                    "chain_rule": "C1",
                    "chain_flag": 0,
                },
            ],
            [
                {
                    "surface": "これ",
                    "pos": "名詞:代名詞:一般:*",
                    "c_type": "*",
                    "c_form": "*",
                    "pron": "コレ",
                    "accent_type": 0,
                    "accent_con_type": "C3",
                    "chain_flag": -1,
                },
                {
                    "surface": "は",
                    "pos": "助詞:係助詞:*:*",
                    "c_type": "*",
                    "c_form": "*",
                    "pron": "ワ",
                    "accent_type": 0,
                    "accent_con_type": "名詞%F1,動詞%F2@0,形容詞%F2@0",
                    "chain_flag": 1,
                },
                {
                    "surface": "テスト",
                    "pos": "名詞:サ変接続:*:*",
                    "c_type": "*",
                    "c_form": "*",
                    "pron": "テスト",
                    "accent_type": 1,
                    "accent_con_type": "C1",
                    "chain_flag": 0,
                },
            ],
        ),
    ]:
        results = convert_njd_feature_to_marine_feature(features)
        assert results == expect


def test_convert_open_jtalk_format_label():
    for labels, morph_boundary, expect in [
        # Fully aligned phrase boundary and word boundary
        (
            {
                "mora": [
                    [
                        "ナ",
                        "ダ",
                        "レ",
                        "デ",
                        "ド",
                        "ー",
                        "ロ",
                        "ガ",
                        "フ",
                        "サ",
                        "ガ",
                        "ッ",
                        "タ",
                        ".",
                    ],
                ],
                "intonation_phrase_boundary": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ],
                "accent_phrase_boundary": [[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                "accent_status": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            },
            [np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)],
            {
                "accent_status": [0, 0, 1, 0, 3, 0, 0],
                "accent_phrase_boundary": [-1, 1, 0, 1, 0, 1, 1],
            },
        ),
        # Not aligned phrase boundary and word boundary (boundary in word)
        (
            {
                "mora": [
                    [
                        "ナ",
                        "ダ",
                        "レ",
                        "デ",
                        "ド",
                        "ー",
                        "ロ",
                        "ガ",
                        "フ",
                        "サ",
                        "ガ",
                        "ッ",
                        "タ",
                        ".",
                    ],
                ],
                "intonation_phrase_boundary": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ],
                "accent_phrase_boundary": [[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]],
                "accent_status": [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
            },
            [np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)],
            {
                "accent_status": [5, 0, 0, 0, 0, 0, 0],
                "accent_phrase_boundary": [-1, 1, 1, 1, 0, 1, 1],
            },
        ),
    ]:
        result = convert_open_jtalk_format_label(labels, morph_boundary)
        assert result == expect


def test_get_accent_nucleus_in_binary_accent_stauts_seq():
    for accents, expect in [
        # for 0 type
        (np.array([0, 0, 0, 0, 0]), 0),
        # for 1 type
        (np.array([1, 0, 0, 0, 0]), 1),
        # for 2 type
        (np.array([0, 1, 0, 0, 0]), 2),
        # for when included multipe ANs
        (np.array([0, 1, 1, 0, 0]), 2),
    ]:
        accent_nucleus_index = get_accent_nucleus_in_binary_accent_stauts_seq(
            accents, binary_accent_nucleus_label=1
        )
        assert accent_nucleus_index == expect


def test_get_accent_nucleus_in_high_low_accent_stauts_seq():
    for accents, expect in [
        # for 0 type
        (np.array([0, 1, 1, 1, 1]), 0),
        # for 1 type
        (np.array([1, 0, 0, 0, 0]), 1),
        # for 2 type
        (np.array([0, 1, 0, 0, 0]), 2),
        # for when included multipe ANs
        (np.array([0, 1, 0, 1, 0]), 2),
    ]:
        accent_nucleus_index = get_accent_nucleus_in_high_low_accent_stauts_seq(
            accents, high_low_accent_nucleus_label=0
        )
        assert accent_nucleus_index == expect


def test_convert_label_by_accent_representation_model():
    for (
        accent,
        accent_phrase_boundary,
        mora,
        current_accent_represent_mode,
        target_accent_represent_mode,
        expect,
    ) in [
        (
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            ["ア", "ア", "ア", "ア", "ア", "ア"],
            "binary",
            "high_low",
            np.array([0, 1, 1, 1, 0, 0]),
        ),
        (
            np.array([0, 1, 1, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            ["ア", "ア", "ア", "ア", "ア", "ア"],
            "high_low",
            "binary",
            np.array([0, 0, 0, 1, 0, 0]),
        ),
        (
            np.array([0, 0, 1, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            ["ア", "ア", "ア", "ア", "ア", "ア"],
            "binary",
            "high_low",
            np.array([0, 1, 1, 1, 0, 0]),
        ),
        (
            np.array([0, 1, 1, 1, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            ["ア", "ア", "ア", "ア", "ア", "ア"],
            "high_low",
            "binary",
            np.array([0, 0, 0, 1, 0, 0]),
        ),
    ]:
        accent = convert_label_by_accent_representation_model(
            accent,
            accent_phrase_boundary,
            mora,
            current_accent_represent_mode,
            target_accent_represent_mode,
            binary_accent_nucleus_label=1,
            high_low_accent_nucleus_label=0,
            accent_phrase_boundary_label=1,
        )

        assert np.all(accent == expect)
