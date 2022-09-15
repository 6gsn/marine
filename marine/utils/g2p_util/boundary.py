from .util import CONNECTABLE_MORA, HALF_PUNCTUATION


def represent_syllable_boundary(index, moras, len_phonemes):
    """
    Represent syllable boundary by 2 types
    - types:
        - 0: non-boundary
        - 1: syllable boundary
    """

    if moras[index] in HALF_PUNCTUATION:
        return [1]

    # Init the boundary into normal boundary
    # e.g., か -> [k, a] -> [0, 1]
    #       わ -> [wa] -> [1]
    boundary = [0] * (len_phonemes - 1) + [1]

    # remove syllable boundaray if next mora is connectable mora (i.e. ッ, ン)
    # e.g., [ ... "デ", "ン" ] -> [ ... [d0, e1], [N1] ] -> [ ... [d0, e0], [N1] ]
    if index + 1 < len(moras):
        if (
            moras[index] not in CONNECTABLE_MORA
            and moras[index + 1] in CONNECTABLE_MORA
        ):
            boundary[-1] = 0
    else:
        boundary[-1] = 1

    return boundary
