UNACCENTED_MORA = "ン"
CONNECTABLE_MORA = set(["ッ", UNACCENTED_MORA])

NON_MORA_LIST = set(["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ"])

LONGVOWEL_CHARACTER = "ー"

FULL_PUNCTUATION = "、。？！"
HALF_PUNCTUATION = ",.?!"

PHON_TABLE = {
    # x
    "ア": ["a"],
    "イ": ["i"],
    "ウ": ["u"],
    "エ": ["e"],
    "オ": ["o"],
    # w
    "ワ": ["w", "a"],
    "ウィ": ["w", "i"],
    "ウェ": ["w", "e"],
    "ウォ": ["w", "o"],
    # y
    "ヤ": ["y", "a"],
    "ユ": ["y", "u"],
    "ヨ": ["y", "o"],
    "イェ": ["y", "e"],
    # k
    "カ": ["k", "a"],
    "キ": ["k", "i"],
    "ク": ["k", "u"],
    "ケ": ["k", "e"],
    "コ": ["k", "o"],
    # kw
    "クヮ": ["kw", "a"],
    "キャ": ["ky", "a"],
    "キュ": ["ky", "u"],
    "キェ": ["ky", "e"],
    "キョ": ["ky", "o"],
    # g
    "ガ": ["g", "a"],
    "ギ": ["g", "i"],
    "グ": ["g", "u"],
    "ゲ": ["g", "e"],
    "ゴ": ["g", "o"],
    # gw
    "グヮ": ["gw", "a"],
    # gy
    "ギャ": ["gy", "a"],
    "ギュ": ["gy", "u"],
    "ギェ": ["gy", "e"],
    "ギョ": ["gy", "o"],
    # s
    "サ": ["s", "a"],
    "スィ": ["s", "i"],
    "ス": ["s", "u"],
    "セ": ["s", "e"],
    "ソ": ["s", "o"],
    # sh
    "シ": ["sh", "i"],
    "シェ": ["sh", "e"],
    "シャ": ["sh", "a"],
    "シュ": ["sh", "u"],
    "ショ": ["sh", "o"],
    # z
    "ザ": ["z", "a"],
    "ズィ": ["z", "i"],
    "ズ": ["z", "u"],
    "ゼ": ["z", "e"],
    "ゾ": ["z", "o"],
    # j
    "ジャ": ["j", "a"],
    "ジ": ["j", "i"],
    "ジュ": ["j", "u"],
    "ジェ": ["j", "e"],
    "ジョ": ["j", "o"],
    # t
    "タ": ["t", "a"],
    "ティ": ["t", "i"],
    "トゥ": ["t", "u"],
    "テ": ["t", "e"],
    "ト": ["t", "o"],
    # ty
    "テャ": ["ty", "a"],
    "テュ": ["ty", "u"],
    "テョ": ["ty", "o"],
    # d
    "ダ": ["d", "a"],
    "ディ": ["d", "i"],
    "ドゥ": ["d", "u"],
    "デ": ["d", "e"],
    "ド": ["d", "o"],
    # dy
    "デャ": ["dy", "a"],
    "デュ": ["dy", "u"],
    "デョ": ["dy", "o"],
    # ch
    "チャ": ["ch", "a"],
    "チ": ["ch", "i"],
    "チュ": ["ch", "u"],
    "チェ": ["ch", "e"],
    "チョ": ["ch", "o"],
    # ts
    "ツァ": ["ts", "a"],
    "ツィ": ["ts", "i"],
    "ツ": ["ts", "u"],
    "ツェ": ["ts", "e"],
    "ツォ": ["ts", "o"],
    # n
    "ナ": ["n", "a"],
    "ニ": ["n", "i"],
    "ヌ": ["n", "u"],
    "ネ": ["n", "e"],
    "ノ": ["n", "o"],
    # ny
    "ニャ": ["ny", "a"],
    "ニュ": ["ny", "u"],
    "ニェ": ["ny", "e"],
    "ニョ": ["ny", "o"],
    # h
    "ハ": ["h", "a"],
    "ヒ": ["h", "i"],
    "ヘ": ["h", "e"],
    "ホ": ["h", "o"],
    # hy
    "ヒャ": ["hy", "a"],
    "ヒュ": ["hy", "u"],
    "ヒェ": ["hy", "e"],
    "ヒョ": ["hy", "ふ"],
    # p
    "パ": ["p", "a"],
    "ピ": ["p", "i"],
    "プ": ["p", "u"],
    "ペ": ["p", "e"],
    "ポ": ["p", "o"],
    # py
    "ピャ": ["py", "a"],
    "ピュ": ["py", "u"],
    "ピェ": ["py", "e"],
    "ピョ": ["py", "o"],
    # b
    "バ": ["b", "a"],
    "ビ": ["b", "i"],
    "ブ": ["b", "u"],
    "ベ": ["b", "e"],
    "ボ": ["b", "o"],
    # by
    "ビャ": ["by", "a"],
    "ビュ": ["by", "u"],
    "ビェ": ["by", "e"],
    "ビョ": ["by", "o"],
    "ヴャ": ["by", "a"],
    "ヴュ": ["by", "u"],
    "ヴョ": ["by", "o"],
    # v
    "ヴァ": ["v", "a"],
    "ヴィ": ["v", "i"],
    "ヴ": ["v", "u"],
    "ヴェ": ["v", "e"],
    "ヴォ": ["v", "o"],
    # f
    "ファ": ["f", "a"],
    "フィ": ["f", "i"],
    "フ": ["f", "u"],
    "フェ": ["f", "e"],
    "フォ": ["f", "o"],
    # m
    "マ": ["m", "a"],
    "ミ": ["m", "i"],
    "ム": ["m", "u"],
    "メ": ["m", "e"],
    "モ": ["m", "o"],
    # my
    "ミャ": ["my", "a"],
    "ミュ": ["my", "u"],
    "ミェ": ["my", "e"],
    "ミョ": ["my", "o"],
    # r
    "ラ": ["r", "a"],
    "リ": ["r", "i"],
    "ル": ["r", "u"],
    "レ": ["r", "e"],
    "ロ": ["r", "o"],
    # ry
    "リャ": ["ry", "a"],
    "リュ": ["ry", "u"],
    "リェ": ["ry", "e"],
    "リョ": ["ry", "o"],
    "ン": ["N"],
    "ッ": ["cl"],
    # for punctuation
    ",": [","],
    ".": ["."],
    "?": ["?"],
    "!": ["!"],
    "、": [", "],
    "。": ["."],
    "？": ["?"],
    "！": ["!"],
}

SUPPORTED_MORA = set(PHON_TABLE.keys())


def get_phoneme(mora, current_phonemes):
    """
    Convert mora(single or double Katakana characters) to Phoneme
    """

    # If the current mora is a long-vowel symbol, add a vowel to the previous phoneme
    # e.g., ワー -> w + a + a -> w + aa
    if current_phonemes and mora == LONGVOWEL_CHARACTER:
        # cl (i.e., ッ) should not be copied from long-vowel (coco #28)
        if current_phonemes[-1] != "cl":
            current_phonemes[-1] = f"{current_phonemes[-1]}{current_phonemes[-1][-1]}"
        phoneme = []
    else:
        try:
            phoneme = PHON_TABLE[mora]
        except KeyError:
            raise ValueError(f"Not supported mora : {mora}")

    return phoneme
