import warnings

import numpy as np
from marine.data.feature.feature_table import RAW_FEATURE_KEYS

OPEN_JTALK_FEATURE_INDEX_TABLE = {
    "surface": 0,
    "pos": [1, 2, 3, 4],
    "c_type": 5,
    "c_form": 6,
    "pron": 9,
    "accent_type": 10,
    "accent_con_type": 11,
    "chain_flag": 12,
}
OPEN_JTALK_FEATURE_RENAME_TABLE = {
    "surface": "string",
    "c_type": "ctype",
    "c_form": "cform",
    "accent_type": "acc",
    "accent_con_type": "chain_rule",
    "chain_flag": "chain_flag",
}

PUNCTUATION_FULL_TO_HALF_TABLE = {
    "、": ",",
    "。": ".",
    "？": "?",
    "！": "!",
}
PUNCTUATION_FULL_TO_HALF_TRANS = str.maketrans(PUNCTUATION_FULL_TO_HALF_TABLE)


def convert_open_jtalk_node_to_feature(nodes):
    features = []
    raw_feature_keys = RAW_FEATURE_KEYS["open-jtalk"]

    for node in nodes:
        # parse feature
        _node = node.split(",")
        node_feature = {}

        for feature_key in raw_feature_keys:
            index = OPEN_JTALK_FEATURE_INDEX_TABLE[feature_key]

            if feature_key == "pos":
                value = ":".join([_node[i] for i in index])
            elif feature_key == "accent_type":
                value = int(_node[index].split("/")[0])
            elif feature_key == "accent_con_type":
                value = _node[index].replace("/", ",")
            elif feature_key == "chain_flag":
                value = int(_node[index])
            elif feature_key == "pron":
                value = _node[index].replace("’", "").replace("ヲ", "オ")
            else:
                value = _node[index]

            node_feature[feature_key] = value

        if node_feature["surface"] == "・":
            continue
        elif node_feature["surface"] in PUNCTUATION_FULL_TO_HALF_TABLE.keys():
            surface = node_feature["surface"].translate(PUNCTUATION_FULL_TO_HALF_TRANS)
            pron = None
            node_feature["surface"] = surface
            node_feature["pron"] = pron

        features.append(node_feature)

    return features


def convert_njd_feature_to_marine_feature(njd_features):
    marine_features = []

    raw_feature_keys = RAW_FEATURE_KEYS["open-jtalk"]
    for njd_feature in njd_features:
        marine_feature = {}
        for feature_key in raw_feature_keys:
            if feature_key == "pos":
                value = ":".join(
                    [
                        njd_feature["pos"],
                        njd_feature["pos_group1"],
                        njd_feature["pos_group2"],
                        njd_feature["pos_group3"],
                    ]
                )
            elif feature_key == "accent_con_type":
                value = njd_feature["chain_rule"].replace("/", ",")
            elif feature_key == "pron":
                value = njd_feature["pron"].replace("’", "").replace("ヲ", "オ")
            else:
                value = njd_feature[OPEN_JTALK_FEATURE_RENAME_TABLE[feature_key]]
            marine_feature[feature_key] = value

        if marine_feature["surface"] == "・":
            continue
        elif marine_feature["surface"] in PUNCTUATION_FULL_TO_HALF_TABLE.keys():
            surface = marine_feature["surface"].translate(
                PUNCTUATION_FULL_TO_HALF_TRANS
            )
            pron = None
            marine_feature["surface"] = surface
            marine_feature["pron"] = pron

        marine_features.append(marine_feature)

    return marine_features


def convert_open_jtalk_format_label(
    labels,
    morph_boundaries,
    accent_nucleus_label=1,
    accent_phrase_boundary_label=1,
    morph_boundary_label=1,
):
    assert "accent_status" in labels.keys(), "`accent_status` is missing in labels"
    assert (
        "accent_phrase_boundary" in labels.keys()
    ), "`accent_phrase_boundary` is missing in labels"

    # squeeze results
    mora_accent_status = labels["accent_status"][0]
    mora_accent_phrase_boundary = labels["accent_phrase_boundary"][0]
    morph_boundary = morph_boundaries[0]

    assert len(mora_accent_status) == len(mora_accent_phrase_boundary), (
        "Not match sequence lenght between"
        "`accent_status`, `morph_boundary`, and `accent_phrase_boundary`"
    )

    mora_accent_phrase_boundary = np.array(mora_accent_phrase_boundary)

    # convert mora-based accent phrase boundary label to morph-based label
    morph_boundary_indexes = np.where(morph_boundary == morph_boundary_label)[0]
    morph_accent_phrase_boundary = np.split(
        mora_accent_phrase_boundary, morph_boundary_indexes
    )
    # `chain_flag` in OpenJTalk represents the status whether the morph will be connected
    morph_accent_phrase_boundary = [
        0 if boundary[0] == accent_phrase_boundary_label else 1
        for boundary in morph_accent_phrase_boundary
    ]
    # first `chain_flag` must be -1
    morph_accent_phrase_boundary[0] = -1
    num_boundary = morph_accent_phrase_boundary.count(0) + 1

    # convert mora-based accent status label to ap-based label
    mora_accent_phrase_boundary_indexes = np.where(
        mora_accent_phrase_boundary == accent_phrase_boundary_label
    )[0]
    phrase_accent_statuses = np.split(
        mora_accent_status, mora_accent_phrase_boundary_indexes
    )
    phrase_accent_status_labels = []

    for phrase_accent_status in phrase_accent_statuses:
        accent_nucleus_indexes = np.where(phrase_accent_status == accent_nucleus_label)[
            0
        ]
        if len(accent_nucleus_indexes) == 0:
            accent_nucleus_index = 0
        else:
            accent_nucleus_index = accent_nucleus_indexes[0] + 1
        phrase_accent_status_labels.append(accent_nucleus_index)

    if len(phrase_accent_status_labels) > num_boundary:
        warnings.warn(
            (
                "Lenght of AP-based accent status will be adjusted "
                "by morph-based accent phrase boundary: "
                f"{len(phrase_accent_status_labels)} > {num_boundary}"
            )
        )
        phrase_accent_status_labels = phrase_accent_status_labels[:num_boundary]

    # convert mora-based accent status to morph-based label
    # the accent label for OpenJTalk pushed in first morph
    morph_accent_status = [
        phrase_accent_status_labels.pop(0) if morph_accent_phrase_flag < 1 else 0
        for morph_accent_phrase_flag in morph_accent_phrase_boundary
    ]

    return {
        "accent_status": morph_accent_status,
        "accent_phrase_boundary": morph_accent_phrase_boundary,
    }
