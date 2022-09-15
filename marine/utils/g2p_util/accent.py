def set_accent_status(accent):
    high = -1
    end_low = -1

    # 1-type : H-L-L ...
    if accent == 1:
        high = 0
        end_low = 1

    # 0-type : L-H-H ...
    elif accent <= 0:
        high = 1

    # N-type : L-H ... H_n-L_n+1-L_n+2 ..
    else:
        high = 1
        end_low = accent

    return high, end_low


def represent_accent_high_low(index, high, end_low):
    """
    Represent the accent by a current status of the mora
    - types:
        - 0: Low
        - 1: High
    """

    # Init accent into flat-accent(=0)
    accent = 0

    # set current accent to high
    if index >= high and (end_low < 0 or index < end_low):
        accent = 1

    return accent


def represent_longvowel_accent_high_low(index, high, end_low):
    # Init accent into flat accent
    accent = 0

    # make previous accent to high
    if index >= high and (end_low < 0 or index <= end_low):
        accent = 1

    return accent


def represent_accent_binary(index, high, end_low):
    """
    Represent the accent by a current status of the mora
    - types:
        - 0: Not accent nucleus
        - 1: Accent nucleus
    """

    # Init accent into flat-accent(=0), and make previous accent to high if match it
    accent = 1 if index == end_low - 1 else 0

    return accent


def represent_longvowel_accent_binary(index, high, end_low):
    # Init accent into a flat accent, and make previous accent to high if match it
    return 1 if index == end_low else 0
