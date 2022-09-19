import json
import random
from logging import getLogger

import numpy as np
import torch
from marine.utils.g2p_util import mora2phon, pron2mora
from marine.utils.regex import has_longvowel

logger = getLogger(__name__)

BINARY_ACCENT_REPRESENT_MODE = "binary"
HIGH_LOW_ACCENT_REPRESENT_MODE = "high_low"
AVAILABLE_ACCENT_REPRESENT_MODES = (
    BINARY_ACCENT_REPRESENT_MODE,
    HIGH_LOW_ACCENT_REPRESENT_MODE,
)

LABEL_TABLE = {
    "intonation_phrase_boundary": {
        "O": 1,
        "B": 2,
    }
}
IGNORE_TARGET_MORA_FOR_ACCENT = ["ー", "ッ"]
JSON_LOG_STATUS_TO_TAG = {True: "success", False: "fail"}
JSON_LOG_TAG_TO_STATUS = {tag: status for status, tag in JSON_LOG_STATUS_TO_TAG.items()}

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


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_corpus(file_path, suffix="json"):
    """Retrieve a list of corpus's path."""
    corpus = []

    if file_path.is_dir():
        file_paths = [path for path in file_path.glob(f"*.{suffix}")]
    else:
        file_paths = [file_path]

    for path in file_paths:
        with open(path, "r") as file:
            corpus += json.load(file)

    return corpus


def split_corpus(
    corpus,
    valid_data_ratio=0.05,
    test_data_ratio=0.05,
    shuffle=True,
    random_state=1,
    absolute_test_size=-1,
):
    try:
        from sklearn.model_selection import train_test_split
    except BaseException:
        raise ImportError('Please install sklearn by `pip install -e ".[dev]"`')

    """Split corpus into train, valid, test."""
    if absolute_test_size > 0:
        assert absolute_test_size < len(corpus)

        train, valid_test = train_test_split(
            corpus,
            test_size=absolute_test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        # split valid/test to half size
        valid, test = train_test_split(
            valid_test,
            test_size=0.5,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        train, valid_test = train_test_split(
            corpus,
            test_size=(valid_data_ratio + test_data_ratio),
            shuffle=shuffle,
            random_state=random_state,
        )
        valid, test = train_test_split(
            valid_test,
            test_size=(test_data_ratio / (valid_data_ratio + test_data_ratio)),
            shuffle=shuffle,
            random_state=random_state,
        )

    return {"train": train, "val": valid, "test": test}


def sequence_mask(lengths, max_len=None):
    """Compute sequence mask."""
    batch_size = lengths.size(0)

    if max_len is None:
        max_len = lengths.max().item()

    ranges = torch.arange(0, max_len, device=lengths.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    lens_exp = lengths.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def pack_inputs(inputs, embedding_keys, device):
    """Covnert batch to tensor for input"""
    embeddings = {key: inputs[key].to(device) for key in embedding_keys}

    inputs = {
        "embedding_features": embeddings,
        "lengths": inputs["mora_length"].cpu(),
        "mask": sequence_mask(inputs["mora_length"]).to(device),
        "prev_decoder_outputs": {},
    }
    return inputs


def pack_outputs(outputs, device):
    """Covnert batch to tensor for output"""
    outputs = {
        task: {
            "label": outputs[task]["label"].to(device),
            "length": outputs[task]["length"].cpu(),
            "mask": sequence_mask(outputs[task]["length"]).to(device),
        }
        for task in outputs.keys()
    }
    return outputs


def pad_incomplete_accent_logits(logits, target_mask):
    """Add pad token for logit with wrong ap length"""
    if logits.size(1) > target_mask.size(1):
        logits = logits[:, : target_mask.size(1), :]
    elif logits.size(1) < target_mask.size(1):
        _logits = torch.zeros(logits.size(0), target_mask.size(1), logits.size(2))
        _logits[:, : logits.size(1), :] = logits
        logits = _logits.to(logits.device)

    return logits


def get_ap_length(accent_phrases, masks, accent_phrase_bondary_label=2):
    """Count number of accent phrase boundary in sequence.
    Args:
        accent_phrases (tensor): The sequence represents accent phrase.
        mask (tensor): The mask for accent phrase boundary sequence.
        accent_phrase_bondary_label (int, optional):
            The label to represent accnet phrase boundary.
    Returns:
        list: A list of lengths for accent phrase boundary.
    """
    lengths = []

    for accent_phrase, mask in zip(accent_phrases, masks):
        accent_phrase = accent_phrase[mask]
        num_boundary = len(accent_phrase[accent_phrase == accent_phrase_bondary_label])
        lengths.append(num_boundary + 1)

    return lengths


def expand_word_label_to_mora(labels, moras, boundaries, target):
    """Convert word-basd label sequence to mora-based label sequence."""
    mora_labels = []

    for label, mora, boundary in zip(labels, moras, boundaries):
        # Remove unavailable label; because jpbp's positive label means that
        # the break comes after the tagged token,
        # i.e., the label located at last token is meanless
        if target == "intonation_phrase_boundary":
            label[-1] = 0

        label = np.array(label)

        boundary_indexs = np.where(boundary > 0)[0]

        assert len(boundary_indexs) + 1 == len(
            label
        ), f"Not matched length: {len(boundary_indexs) + 1} != {len(label)}"

        if target == "intonation_phrase_boundary":
            label_indexs = np.where(label > 0)[0]
        else:
            raise NotImplementedError(f"Not supported error: {target}")

        # annotate break label
        _label = np.array([LABEL_TABLE[target]["O"]] * len(mora))
        _label[boundary_indexs[label_indexs]] = LABEL_TABLE[target]["B"]

        mora_labels.append(list(_label))

    return mora_labels


def _convert_ap_based_accent_to_mora_based_accent(
    ap_accents,
    phrases,
    mode=HIGH_LOW_ACCENT_REPRESENT_MODE,
    mora=None,
    accent_phrase_boundary_label=2,
):
    """Convert accent phrase-based accent status sequence to mora-based."""
    assert mode in AVAILABLE_ACCENT_REPRESENT_MODES, (
        f"Not supported representation mode {mode}:",
        "Representation mode must be selected in binary and high_low",
    )

    ap_accents = ap_accents.cpu()
    phrases = phrases.cpu()

    boundaries = np.where(phrases == accent_phrase_boundary_label)[0]
    phrases = np.split(phrases, boundaries)

    if mora is not None:
        ap_moras = np.split(np.array(mora), boundaries)

    # Pad or slice if there is miss-macth between predicted phrase and real-label
    if len(phrases) < len(ap_accents):
        ap_accents = ap_accents[: len(phrases)]
    elif len(phrases) > len(ap_accents):
        ap_accents = np.append(ap_accents, [0] * (len(phrases) - len(ap_accents)))

    assert len(phrases) == len(
        ap_accents
    ), f"Not matched seq lengths {len(phrases)} != {len(ap_accents)}"

    mora_accents = []

    for index, (accent, phrase) in enumerate(zip(ap_accents, phrases)):
        if mode == BINARY_ACCENT_REPRESENT_MODE:
            if mora is not None:
                moras = ap_moras[index]
            else:
                moras = ["*"] * len(phrase)

            # convert to 0-based label to index
            accent_label = accent.item() - 2
            mora_accent = [0] * len(phrase)

            if (
                len(mora_accent) > accent_label >= 0
                and moras[accent_label] not in IGNORE_TARGET_MORA_FOR_ACCENT
            ):
                mora_accent[accent_label] = 1
        elif mode == HIGH_LOW_ACCENT_REPRESENT_MODE:
            if mora is not None:
                moras = ap_moras[index]
            else:
                moras = ["*"] * len(phrase)

            # convert to 0-based label to index
            accent_label = accent.item() - 1

            # ignore accent located at out of mora seq
            if accent_label > len(moras):
                accent_label = 0
            # ignore accent located at long-vowel
            elif (
                accent_label > 0
                and moras[accent_label - 1] in IGNORE_TARGET_MORA_FOR_ACCENT
            ):
                accent_label = 0

            _, mora_accent = pron2mora(moras, accent_label, mode)
        else:
            raise NotImplementedError(
                (
                    f"Not supported representation mode {mode}:"
                    " Representation mode must be selected in binary and high_low"
                )
            )

        mora_accents += mora_accent

    return np.array(mora_accents)


def convert_ap_based_accent_to_mora_based_accent(
    accent_statuses,
    accent_phrase_boundaries,
    ap_seq_masks,
    mora_seq_masks,
    accent_represent_mode=BINARY_ACCENT_REPRESENT_MODE,
):
    """Convert accent phrase-based accent status sequence to mora-based."""
    mora_accent_statuses = np.array([], dtype=np.int64)

    for accent_status, accent_phrase_boundary, ap_seq_mask, mora_seq_mask in zip(
        accent_statuses,
        accent_phrase_boundaries,
        ap_seq_masks,
        mora_seq_masks,
    ):
        # accent_status: (T_ap) -> (T_ap - mask_ap)
        # ap_seq_mask: (T_ap)
        accent_status = accent_status[ap_seq_mask]

        # accent_phrase_boundary: (T_mora) -> (T_mora, mask_mora)
        # mora_seq_mask: (T_mora)
        accent_phrase_boundary = accent_phrase_boundary[mora_seq_mask]

        # convert to mora-based sequence
        mora_accent_status = _convert_ap_based_accent_to_mora_based_accent(
            accent_status, accent_phrase_boundary, mode=accent_represent_mode
        )

        mora_accent_statuses = np.concatenate(
            [mora_accent_statuses, mora_accent_status]
        )

    return mora_accent_statuses


def get_accent_nucleus_in_binary_accent_stauts_seq(
    ap_based_clipped_accents, binary_accent_nucleus_label=2
):
    accent_nucleus_index = np.where(
        ap_based_clipped_accents == binary_accent_nucleus_label
    )[0]
    if len(accent_nucleus_index) >= 1:  # Type: 1 ~ N
        # an accent phrase has at most one accent nucleus
        # and the `accent_nucleus_index` must be represented as 1-based label
        # (for 0 = no accent nucleus)
        accent_nucleus_index = int(accent_nucleus_index[0]) + 1
    else:
        accent_nucleus_index = 0  # Type: 0 (No accent)
    return accent_nucleus_index


def get_accent_nucleus_in_high_low_accent_stauts_seq(
    ap_based_clipped_accents, high_low_accent_nucleus_label=1
):
    low_pitch_locations = list(
        np.where(ap_based_clipped_accents == high_low_accent_nucleus_label)[0]
    )
    if len(low_pitch_locations) > 1:  # Type: 1 ~ N
        low_pitch_pointer = 0 if low_pitch_locations[0] != 0 else 1
        accent_nucleus_index = int(low_pitch_locations[low_pitch_pointer])
    else:
        accent_nucleus_index = 0  # Type: 0 (No accent)
    return accent_nucleus_index


def convert_label_by_accent_representation_model(
    mora_based_accents,
    accent_phrase_boundary,
    moras,
    current_accent_represent_mode,
    target_accent_represent_mode,
    binary_accent_nucleus_label=2,
    high_low_accent_nucleus_label=1,
    accent_phrase_boundary_label=2,
):
    """Convert accent label for following with target accent representation mode"""
    assert current_accent_represent_mode != target_accent_represent_mode
    assert current_accent_represent_mode in AVAILABLE_ACCENT_REPRESENT_MODES
    assert target_accent_represent_mode in AVAILABLE_ACCENT_REPRESENT_MODES

    if isinstance(mora_based_accents, torch.Tensor):
        mora_based_accents = mora_based_accents.cpu()
    elif not isinstance(mora_based_accents, np.ndarray):
        raise TypeError("mora_based_accents must be tensor or numpy.array")

    if isinstance(accent_phrase_boundary, torch.Tensor):
        accent_phrase_boundary = accent_phrase_boundary.cpu()
    elif not isinstance(accent_phrase_boundary, np.ndarray):
        raise TypeError("accent_phrase_boundary must be tensor or numpy.array")

    assert len(mora_based_accents) == len(accent_phrase_boundary) == len(moras)

    converted_accents = []
    accent_phrse_boundary_indexs = np.where(
        accent_phrase_boundary == accent_phrase_boundary_label
    )[0]

    ap_based_clipped_moras = np.split(moras, accent_phrse_boundary_indexs)
    ap_based_clipped_accents = np.split(
        mora_based_accents, accent_phrse_boundary_indexs
    )

    for ap_based_clipped_mora, ap_based_clipped_accent in zip(
        ap_based_clipped_moras, ap_based_clipped_accents
    ):
        if current_accent_represent_mode == HIGH_LOW_ACCENT_REPRESENT_MODE:
            accent_nucleus_index = get_accent_nucleus_in_high_low_accent_stauts_seq(
                ap_based_clipped_accent, high_low_accent_nucleus_label
            )
        elif current_accent_represent_mode == BINARY_ACCENT_REPRESENT_MODE:
            accent_nucleus_index = get_accent_nucleus_in_binary_accent_stauts_seq(
                ap_based_clipped_accent, binary_accent_nucleus_label
            )
        else:
            raise ValueError(
                f"Not supported accent mode: {current_accent_represent_mode}"
            )

        _, converted_accent = pron2mora(
            ap_based_clipped_mora,
            accent=accent_nucleus_index,
            represent_mode=target_accent_represent_mode,
        )

        converted_accents += converted_accent

    return np.array(converted_accents)


def convert_mora_jp_to_en(mora):
    phonemes, _, syllable_boundary = mora2phon(mora, [0] * len(mora))
    temp_en_mora = ""
    en_moras = []

    for phoneme, boundary in zip(phonemes, syllable_boundary):
        if phoneme in [".", ",", "?", "!", "N", "cl"]:
            if temp_en_mora:
                en_moras.append(temp_en_mora)
                temp_en_mora = ""
            en_moras.append(phoneme)
            continue

        temp_en_mora += phoneme

        if boundary == 1:
            if has_longvowel(temp_en_mora):
                consonant = temp_en_mora[:-1]
                vowel = temp_en_mora[-1]
                en_moras += [consonant, vowel]
            else:
                en_moras.append(temp_en_mora)
            temp_en_mora = ""

    return en_moras


def plot_attention(attention, xs=None, ys=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except BaseException:
        raise ImportError('Please install matplotlib by `pip install -e ".[dev]"`')

    fig, ax = plt.subplots()
    attention = attention.cpu().data.numpy().T

    # draw attention
    im = ax.imshow(attention, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)

    # set labels
    if xs:
        ax.set_xticks(list(range(len(xs))))
        ax.set_xticklabels(xs)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    if ys:
        ax.set_yticks(list(range(len(ys))))
        ax.set_yticklabels(ys)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return fig


def plot_batch_attention(
    inputs,
    logits,
    ap_lengths,
    attentions,
    feature_set,
    plot_targets=None,
    tensorboard_writer=None,
    phase=None,
    epoch=None,
    script_ids=None,
    accent_phrase_boundary_label=2,
):
    """Plot attentions in a batch"""
    # parse features
    moras = inputs["embedding_features"]["mora"]
    input_lengths = inputs["lengths"]
    aps = inputs["prev_decoder_outputs"]["accent_phrase_boundary"].cpu().data.numpy()

    if plot_targets:
        indexs = plot_targets
    else:
        indexs = range(len(attentions))

    for index in indexs:
        # skip plotting when got invalid index
        if index >= len(attentions):
            continue

        attention = attentions[index][: ap_lengths[index], : input_lengths[index]]
        accents = torch.argmax(logits[index][: ap_lengths[index]], dim=1).tolist()
        ap = aps[index][: input_lengths[index]].tolist()
        mora = feature_set.convert_id_to_feature(
            "mora", moras[index][: input_lengths[index]].tolist()
        )
        mora = convert_mora_jp_to_en(mora)

        xs = [str(a - 1) for a in accents]
        ys = [
            f"{m}{'/' if a == accent_phrase_boundary_label else '_'}"
            for m, a in zip(mora, ap)
        ]

        fig = plot_attention(attention, xs=xs, ys=ys)

        if tensorboard_writer:
            tensorboard_writer.add_figure(
                f"{phase}/attention/{script_ids[index]}", fig, epoch
            )


def convert_readable_labels(predicts, targets, masks, script_ids):
    """Convert logits to readable label"""
    logs = {}

    for predict, target, mask, script_id in zip(predicts, targets, masks, script_ids):
        predict = [str(int(y)) for y in predict[mask].cpu().tolist()]
        target = [str(int(y)) for y in target[mask].cpu().tolist()]

        logs[script_id] = {
            "predict": ",".join(predict),
            "target": ",".join(target),
            "status": JSON_LOG_STATUS_TO_TAG[predict == target],
        }

    return logs


def group_by_script_id(logs):
    _logs = {}

    for task, scripts in logs.items():
        for script_id, script_info in scripts.items():
            if script_id not in _logs.keys():
                _logs[script_id] = {task: script_info}
            else:
                _logs[script_id][task] = script_info

    return _logs


def _make_task_group_variation(tasks, min_num=2):
    groups = []
    target_size = len(tasks)
    while target_size - min_num >= 0:
        for x in range(0, (len(tasks) - target_size) + 1):
            group = tasks[x : x + target_size]
            groups.append(group)
        target_size -= 1
    return groups


def _calculate_multiple_task_scores(tasks, logs):
    task_groups = _make_task_group_variation(tasks)
    multiple_task_scores = {}

    for task_group in task_groups:
        multiple_task_key = "+".join(task_group)
        statuses = np.array(
            [
                sum(
                    [
                        JSON_LOG_TAG_TO_STATUS[script_info[task]["status"]]
                        for task in task_group
                    ]
                )
                for script_info in logs.values()
            ]
        )
        num_all_task_success = len(np.where(statuses >= len(task_group))[0])
        multiple_task_scores[multiple_task_key] = {
            "sentence_level_accuracy": num_all_task_success / len(logs)
        }

    return multiple_task_scores


def log_scores(
    phase,
    epoch,
    tasks,
    metrics,
    logs=None,
    loss=None,
    tensorboard_writer=None,
):
    """Log scores"""

    scores = metrics.compute()

    # merge loss into score metrics
    if loss:
        for task in tasks:
            scores[task]["loss"] = loss[task].item()

    if logs:
        multiple_task_scores = _calculate_multiple_task_scores(tasks, logs)
        tasks += list(multiple_task_scores.keys())
        scores.update(multiple_task_scores)

    for task in tasks:
        # logging to shell
        score_log = f"{phase} / {task} |" + " |".join(
            [
                f" {score_name} : {score:.4f}"
                if score_name == "loss"
                else f" {score_name} : {score:.4%}"
                for score_name, score in scores[task].items()
            ]
        )
        logger.info(score_log)

        # logging to tensorboard
        if tensorboard_writer:
            for score_name, score in scores[task].items():
                tensorboard_writer.add_scalar(
                    f"{phase}/{task}/{score_name}", score, epoch
                )
