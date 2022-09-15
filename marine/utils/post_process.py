import re
from csv import reader

import numpy as np
from marine.utils.g2p_util import pron2mora
from marine.utils.g2p_util.g2p import ACCENT_REPRESENT_FUNC_TABLE

FEATURE_PARSE_SYMBOL = "/"
FEATURE_NODE_SPLIT_SYMBOL = "_"
FEATURE_MORA_SPLIT_SYMBOL = ","
FEATURE_ACCENT_SYMBOL = "@"

FEATURE_SYOMBOL_REMOVER = str.maketrans(
    {
        FEATURE_NODE_SPLIT_SYMBOL: "",
        FEATURE_MORA_SPLIT_SYMBOL: "",
        FEATURE_ACCENT_SYMBOL: "",
    }
)

PADDING_VALUE_FOR_LABEL = 1


def _make_align_array(surfaces):
    aligns = []
    index = 0

    while index < len(surfaces):
        current_surface = surfaces[index]
        blank_len = len(current_surface) - 1

        boundary = [index + 1]
        blank = [0] * blank_len if blank_len >= 1 else []

        aligns = aligns + boundary + blank

        index += 1

    return aligns


def _is_available_match(aligns, head, tail):
    return (head >= 0 and aligns[head] > 0) and (
        (tail == len(aligns)) or (tail < len(aligns) and aligns[tail] != 0)
    )


def _search_mark(padded_aligns):
    while len(padded_aligns) > 0:
        mark = padded_aligns.pop(-1)
        if mark > 0:
            return mark
    return -1


def aligns2mask(aligns, head, tail):
    if _is_available_match(aligns, head, tail):
        start = aligns[head] - 1
        end = _search_mark(aligns[head:tail])
        return (start, end)
    else:
        return None


def convert_feature_to_value(target, pron, label):
    if target == "accent_status":
        moras = pron2mora(pron)
        value = {}

        for accent_represent_mode in ACCENT_REPRESENT_FUNC_TABLE.keys():
            _, represented_accent = pron2mora(moras, label, accent_represent_mode)
            value[accent_represent_mode] = represented_accent
    else:
        moras = pron2mora(pron)
        value = len(moras) * [0]

        if label > 1:
            value[label - 1] = 1

    return moras, value


def load_postprocess_vocab(vocab_dir, tasks):
    vocab = {key: {} for key in tasks}

    for dict_dir in vocab_dir.iterdir():
        target = dict_dir.name

        if target == "vocab.pkl":
            continue

        assert target in tasks

        for dict_path in dict_dir.glob("*.tsv"):
            with dict_path.open() as dict_file:
                table = reader(dict_file, delimiter="\t")

                for pattern, value in table:
                    regex = re.compile(pattern)
                    surface, feature = value.split(FEATURE_PARSE_SYMBOL)
                    surfaces = surface.split(FEATURE_NODE_SPLIT_SYMBOL)
                    pron = feature.translate(FEATURE_SYOMBOL_REMOVER)

                    features = [
                        [mora for mora in moras.split(FEATURE_MORA_SPLIT_SYMBOL)]
                        for moras in feature.split(FEATURE_NODE_SPLIT_SYMBOL)
                    ]

                    assert len(surfaces) == len(
                        features
                    ), f"Wrong length entry : ({surfaces} != {features})"

                    labels = [
                        (0 if node_index == 0 else len(features[node_index - 1]))
                        + mora_index
                        + 1
                        for node_index, moras in enumerate(features)
                        for mora_index, mora in enumerate(moras)
                        if mora.endswith(FEATURE_ACCENT_SYMBOL)
                    ]

                    if labels:
                        # only use first appeared symbol
                        label = labels[0]
                    else:
                        label = -1

                    moras, values = convert_feature_to_value(target, pron, label)
                    vocab[target][pattern] = (regex, moras, values)

    return vocab


def apply_postprocess_dict(
    task,
    nodes,
    labels,
    moras,
    boundary,
    postprocess_targets,
    postprocess_vocab,
    accent_represent_mode="binary",
):
    surfaces = [node["surface"] for node in nodes]
    surface = "".join(surfaces)

    targets = postprocess_targets.findall(surface)

    if targets:
        aligns = _make_align_array(surfaces)

        for target in targets:
            regex, pron, values = postprocess_vocab[target]

            for match in regex.finditer(surface):
                head, tail = match.span()

                node_mask = aligns2mask(aligns, head, tail)

                if node_mask:
                    # get mora-based boundary's position
                    boundary_indexs = (
                        [0] + list(np.where(boundary > 0)[0]) + [len(moras)]
                    )
                    node_start, node_end = node_mask

                    mora_mask = slice(
                        boundary_indexs[node_start],
                        boundary_indexs[node_end],
                    )

                    if moras[mora_mask] == pron:
                        if task == "accent_status":
                            value = values[accent_represent_mode]
                        else:
                            value = values

                        labels[mora_mask] = value

    return labels
