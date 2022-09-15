import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from marine.logger import getLogger
from marine.utils.g2p_util import pron2mora
from tqdm import tqdm

logger = None

UNUSED_SYMBOL_REMOVER = str.maketrans("", "", "^$[?")
ACCENT_NUCLEUS_SYMBOL = "]"
ACCENT_PHRASE_BOUNDARY_SYMBOL = "#"
INTONATION_PHRASE_BOUNDARY_SYMBOL = "_"
INTONATION_PHRASE_BOUNDARY_PUNCTUATION = ","


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Special format txt format data to json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_path", type=Path, help="Input path or directory")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--accent_status_seq_level",
        "-s",
        type=str,
        choices=["ap", "mora"],
        default="ap",
        help="Sequence level for accent status label",
    )
    parser.add_argument(
        "--accent_status_represent_mode",
        "-m",
        type=str,
        choices=["binary", "high_low"],
        default="binary",
        help="""Representation mode for accent status label
        (this option will be ignored when --accent-status-seq-level is chosen as 'ap')""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def alignment_feature(features, target_feature, ignore_features):
    # filtering
    for ignore_feature in ignore_features:
        features = features[features != ignore_feature]

    # detect
    _features = features == target_feature

    # move & pad
    mask = ~(np.concatenate([_features[1:], [False]]))
    _features = _features[mask]

    if target_feature in [
        ACCENT_PHRASE_BOUNDARY_SYMBOL,
        INTONATION_PHRASE_BOUNDARY_SYMBOL,
    ]:
        _features = np.concatenate([[False], _features[:-1]])

    return _features


def convert_mask_seq_to_int_seq(feature):
    return np.where(feature, 1, 0)


def merge_ip_ap_boundary(accent_phrase_boundaries, intonation_phrase_boundaries):
    assert len(accent_phrase_boundaries) == len(intonation_phrase_boundaries)

    return accent_phrase_boundaries + intonation_phrase_boundaries


def binary_accent_to_ap_accent(
    binary_accents,
    accent_phrase_boundaries,
    accent_label=1,
    accnet_phrase_label=1,
):
    assert len(binary_accents) == len(accent_phrase_boundaries)

    accent_phrase_boundary_indexes = np.where(
        accent_phrase_boundaries == accnet_phrase_label
    )[0]
    splitted_binary_accents = np.split(binary_accents, accent_phrase_boundary_indexes)

    ap_accent_labbels = []

    for binary_accent in splitted_binary_accents:
        accent_indexs = np.where(binary_accent == accent_label)[0]

        if len(accent_indexs) >= 1:
            acc = accent_indexs[0] + 1  # zero pad
        else:
            acc = 0

        ap_accent_labbels.append(acc)

    return ap_accent_labbels


def binary_accent_to_high_low_accent(
    moras,
    binary_accents,
    accent_phrase_boundaries,
    accent_label=1,
    accnet_phrase_label=1,
    accent_status_represent_mode="high_low",
):
    assert len(moras) == len(binary_accents) == len(accent_phrase_boundaries)

    accent_phrase_boundary_indexes = np.where(
        accent_phrase_boundaries == accnet_phrase_label
    )[0]
    splitted_moras = np.split(moras, accent_phrase_boundary_indexes)
    splitted_binary_accents = np.split(binary_accents, accent_phrase_boundary_indexes)

    high_low_accent_labels = []

    for mora, binary_accent in zip(splitted_moras, splitted_binary_accents):
        accent_indexs = np.where(binary_accent == accent_label)[0]

        if len(accent_indexs) >= 1:
            _acc = accent_indexs[0] + 1  # zero pad
            _, acc = pron2mora(mora, int(_acc), accent_status_represent_mode)
            if accent_status_represent_mode == "high_low" and mora[int(_acc)] == "ー":
                acc[int(_acc)] = 0
        else:
            _, acc = pron2mora(mora, 0, accent_status_represent_mode)

        high_low_accent_labels += acc

    return high_low_accent_labels


def convert_to_srt_feature(features, splitter=","):
    if isinstance(features, np.ndarray):
        features = features.tolist()
    elif not isinstance(features, list):
        raise TypeError("Wrong type of feature")

    return splitter.join([str(value) for value in features])


def parse_jsut_annotation(
    annotation, accent_status_seq_level, accent_status_represent_mode
):
    features = {}

    # preprocessing: remove unused symbols
    annotation = annotation.translate(UNUSED_SYMBOL_REMOVER)
    # preprocessing: replace ヲ -> オ
    annotation = annotation.replace("ヲ", "オ")
    # preprocessing: parse as sequence
    mora_based_annotation = np.array(pron2mora(annotation))

    # filtering symbol
    moras = mora_based_annotation[
        (mora_based_annotation != ACCENT_NUCLEUS_SYMBOL)
        & (mora_based_annotation != ACCENT_PHRASE_BOUNDARY_SYMBOL)
        & (mora_based_annotation != INTONATION_PHRASE_BOUNDARY_SYMBOL)
    ]

    assert len(moras) > 0, "Empty annotation"

    binary_accents = alignment_feature(
        mora_based_annotation,
        ACCENT_NUCLEUS_SYMBOL,
        ignore_features=ACCENT_PHRASE_BOUNDARY_SYMBOL
        + INTONATION_PHRASE_BOUNDARY_SYMBOL,
    )
    accent_phrase_boundaries = alignment_feature(
        mora_based_annotation,
        ACCENT_PHRASE_BOUNDARY_SYMBOL,
        ignore_features=ACCENT_NUCLEUS_SYMBOL + INTONATION_PHRASE_BOUNDARY_SYMBOL,
    )
    intonation_phrase_boundaries = alignment_feature(
        mora_based_annotation,
        INTONATION_PHRASE_BOUNDARY_SYMBOL,
        ignore_features=ACCENT_NUCLEUS_SYMBOL + ACCENT_PHRASE_BOUNDARY_SYMBOL,
    )

    assert (
        len(moras)
        == len(binary_accents)
        == len(accent_phrase_boundaries)
        == len(intonation_phrase_boundaries)
    ), "{} != {} != {} != {}".format(
        len(moras),
        len(binary_accents),
        len(accent_phrase_boundaries),
        len(intonation_phrase_boundaries),
    )

    accents = convert_mask_seq_to_int_seq(binary_accents)
    accent_phrase_boundaries = convert_mask_seq_to_int_seq(accent_phrase_boundaries)
    intonation_phrase_boundaries = convert_mask_seq_to_int_seq(
        intonation_phrase_boundaries
    )

    accent_phrase_boundaries = merge_ip_ap_boundary(
        accent_phrase_boundaries, intonation_phrase_boundaries
    )

    if accent_status_seq_level == "ap":
        accents = binary_accent_to_ap_accent(accents, accent_phrase_boundaries)
    elif accent_status_seq_level == "mora" and accent_status_represent_mode != "binary":
        accents = binary_accent_to_high_low_accent(
            moras,
            accents,
            accent_phrase_boundaries,
            accent_status_represent_mode=accent_status_represent_mode,
        )

    features["pron"] = convert_to_srt_feature(moras, splitter="")
    features["accent_status"] = convert_to_srt_feature(accents)
    features["accent_phrase_boundary"] = convert_to_srt_feature(
        accent_phrase_boundaries
    )
    features["intonation_phrase_boundary"] = convert_to_srt_feature(
        intonation_phrase_boundaries
    )

    return features


def load_jsut_corpus(
    jsut_corpus_dir, accent_status_seq_level, accent_status_represent_mode
):
    text_yaml_path = jsut_corpus_dir / "text_kana" / "basic5000.yaml"
    annotation_yaml_path = jsut_corpus_dir / "e2e_symbol" / "katakana.yaml"

    scripts = []
    texts = {}
    annotations = {}

    with open(text_yaml_path, "r") as file:
        texts = yaml.safe_load(file)

    with open(annotation_yaml_path, "r") as file:
        annotations = yaml.safe_load(file)

    assert texts.keys() == annotations.keys(), "Not matched text and annotations"

    for script_id in tqdm(texts.keys(), "Parse anntoations"):
        features = {}

        surface = texts[script_id]["text_level0"]
        annotation = annotations[script_id]
        feature = parse_jsut_annotation(
            annotation, accent_status_seq_level, accent_status_represent_mode
        )

        features["script_id"] = script_id
        features["surface"] = surface
        features.update(feature)

        scripts.append(features)

    logger.info(f"Loaded {len(scripts)} scripts")

    return scripts


def entry(argv=sys.argv):
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")

    scripts = load_jsut_corpus(
        args.in_path, args.accent_status_seq_level, args.accent_status_represent_mode
    )

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    today = datetime.date.today().strftime("%y%m%d")
    with open(args.out_dir / f"just_corpus_{today}.json", "w") as file:
        json.dump(scripts, file, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == "__main__":
    sys.exit(entry())
