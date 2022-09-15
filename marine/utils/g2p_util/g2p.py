from .accent import (
    represent_accent_binary,
    represent_accent_high_low,
    represent_longvowel_accent_binary,
    represent_longvowel_accent_high_low,
    set_accent_status,
)
from .boundary import represent_syllable_boundary
from .util import (
    CONNECTABLE_MORA,
    HALF_PUNCTUATION,
    LONGVOWEL_CHARACTER,
    NON_MORA_LIST,
    SUPPORTED_MORA,
    get_phoneme,
)

ACCENT_REPRESENT_FUNC_TABLE = {
    "binary": {
        "represent_accent": represent_accent_binary,
        "represent_longvowel_accent": represent_longvowel_accent_binary,
    },
    "high_low": {
        "represent_accent": represent_accent_high_low,
        "represent_longvowel_accent": represent_longvowel_accent_high_low,
    },
}


def pron2mora(pron, accent=None, represent_mode="binary"):
    moras = []
    i = 0

    while i < len(pron):
        current_pron = pron[i]

        if current_pron in NON_MORA_LIST and len(moras) > 0:
            merged_mora = f"{moras[-1]}{current_pron}"

            if merged_mora in SUPPORTED_MORA:
                moras[-1] = merged_mora
            else:
                moras.append(current_pron)
        else:
            moras.append(current_pron)

        i += 1

    if accent is not None:
        if not isinstance(accent, int):
            raise TypeError(f"Accent is must be int not {type(accent)}")

        if represent_mode not in ACCENT_REPRESENT_FUNC_TABLE.keys():
            raise NotImplementedError(f"Not Implemented mode : {represent_mode}")

        # init rule
        accent_rule = ACCENT_REPRESENT_FUNC_TABLE[represent_mode]
        represent_accent = accent_rule["represent_accent"]
        represent_longvowel_accent = accent_rule["represent_longvowel_accent"]

        # init satus
        high, end_low = set_accent_status(accent)
        represented_accents = []

        for index, mora in enumerate(moras):
            # if currnet mora is long-vowel syombol, update last mora
            if _is_longvowel(mora):
                represented_accent = represent_longvowel_accent(index, high, end_low)
                represented_accents.append(represented_accent)
            else:
                represented_accent = represent_accent(index, high, end_low)
                represented_accents.append(represented_accent)

        assert len(moras) == len(
            represented_accents
        ), f"Wrong repersentation : {moras} != {represented_accents}"

        return moras, represented_accents

    return moras


def _is_longvowel(mora):
    return mora == LONGVOWEL_CHARACTER


# Consider whether the mora is long-vowel
# and previous mora was not unaccneted mora for escapte excepted case
# e.g., ンー = NN11, ッー = cl11
def _is_prev_mora_not_unaccented_mora(index, moras):
    return index > 0 and moras[index - 1] not in CONNECTABLE_MORA


def mora2phon(
    moras,
    accents=None,
    ignore_longvowel_accent=False,
    use_punctuation=True,
    punctuation_accent_label=7,
):
    phonemes = []

    if accents is None:
        for mora in moras:
            if not use_punctuation and mora in HALF_PUNCTUATION:
                continue

            phoneme = get_phoneme(mora, phonemes)
            phonemes += phoneme

        return phonemes

    else:
        if len(accents) != len(moras):
            raise ValueError(
                f"Accent is must be same to length of mora (got : {len(accents) != len(moras)})"
            )

        represented_accents = []
        represented_boundaries = []

        for index, (mora, accent) in enumerate(zip(moras, accents)):
            if not use_punctuation and mora in HALF_PUNCTUATION:
                continue

            phoneme = get_phoneme(mora, phonemes)
            len_phoneme = len(phoneme)
            represented_boundary = represent_syllable_boundary(
                index, moras, len_phoneme
            )
            phonemes += phoneme

            # if currnet mora is long-vowel syombol, update last mora
            if _is_longvowel(mora):
                # if currnet mora is long-vowel symbol, only update previous accent
                if not ignore_longvowel_accent and _is_prev_mora_not_unaccented_mora(
                    index, moras
                ):
                    represented_accents[-1] = accent
                represented_boundaries[-1] = represented_boundary[-1]
            elif mora in HALF_PUNCTUATION:
                represented_accent = [punctuation_accent_label]
                represented_accents += represented_accent
                represented_boundaries += represented_boundary
            else:
                represented_accent = [0] * (len_phoneme - 1) + [accent]
                represented_accents += represented_accent
                represented_boundaries += represented_boundary

        return phonemes, represented_accents, represented_boundaries


def pron2phon(pron, accent=None, represent_mode="binary"):
    if represent_mode not in ACCENT_REPRESENT_FUNC_TABLE.keys():
        raise NotImplementedError(f"Not Implemented mode : {represent_mode}")

    if accent is None:
        moras = pron2mora(pron, accent, represent_mode)
        accents = None
    else:
        moras, accents = pron2mora(pron, accent, represent_mode)

    return mora2phon(moras, accents)
