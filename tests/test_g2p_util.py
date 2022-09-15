from logging import getLogger

from marine.utils.g2p_util.accent import set_accent_status
from marine.utils.g2p_util.g2p import pron2mora, pron2phon

logger = getLogger("test")


def test_accent_type_prediction():
    for accent, expected in [
        # Type 0
        (0, (1, -1)),
        # Type 1
        (1, (0, 1)),
        # Type 2
        (2, (1, 2)),
        # Type 3
        (3, (1, 3)),
    ]:

        status = set_accent_status(accent)
        assert status == expected


def test_g2p_as_high_low():
    for index, (pron, accent, expected) in enumerate(
        [
            # For Basic Pattern
            ("ソレデワ", 3, "s00 o01 r00 e11 d00 e11 w00 a01"),
            ("カジノオ", 1, "k00 a11 j00 i01 n00 o01 o01"),
            # For ッ
            ("オクッテイル", 0, "o01 k00 u10 cl11 t00 e11 i11 r00 u11"),
            ("ハジマッタ", 0, "h00 a01 j00 i11 m00 a10 cl11 t00 a11"),
            ("ギインリッポー", 4, "g00 i01 i10 N11 r00 i10 cl01 p00 oo01"),
            # For long Vowel(ー)
            ("シンデイル", 0, "sh00 i00 N11 d00 e11 i11 r00 u11"),
            ("ユキオンナオ", 3, "y00 u01 k00 i11 o10 N01 n00 a01 o01"),
            ("ツーワリョーガ", 3, "ts00 uu11 w00 a11 ry00 oo01 g00 a01"),
            ("アソンデイル", 0, "a01 s00 o10 N11 d00 e11 i11 r00 u11"),
            ("トーナンアジアオ", 5, "t00 oo11 n00 a10 N11 a11 j00 i01 a01 o01"),
            ("サイセーシマス", 6, "s00 a01 i11 s00 ee11 sh00 i11 m00 a11 s00 u01"),
            (
                "トーゴーガタリゾートノ",
                8,
                "t00 oo11 g00 oo11 g00 a11 t00 a11 r00 i11 z00 oo11 t00 o01 n00 o01",
            ),
            # For ン(N)
            ("ンダモシタン", 4, "N01 d00 a11 m00 o11 sh00 i11 t00 a00 N01"),
            ("チョーセンスル", 0, "ch00 oo11 s00 e10 N11 s00 u11 r00 u11"),
            ("ヘンコーオ", 0, "h00 e00 N11 k00 oo11 o11"),
            # For long vowel + ン(N)
            ("バチーンッテ", 2, "b00 a01 ch00 ii10 N01 cl01 t00 e01"),
            ("バーンアウトワ", 4, "b00 aa10 N11 a11 u01 t00 o01 w00 a01"),
            # For ンー(NN)
            ("ンー", 1, "NN11"),
            ("ンー", 0, "NN01"),
            ("ジャンケンー", 1, "j00 a10 N01 k00 e00 NN01"),
            ("オニーチャンー", 2, "o01 n00 ii11 ch00 a00 NN01"),
            # For ッー(cl)
            ("ッー", 1, "cl11"),
            ("ッー", 0, "cl01"),
        ]
    ):
        phons, accents, boundaries = pron2phon(pron, accent, represent_mode="high_low")

        phonemes = " ".join(
            [f"{p}{a}{b}" for p, a, b in zip(phons, accents, boundaries)]
        )

        logger.info(f"No.{index} {pron} / {accent}")
        logger.info(f"answer\t:\t{phonemes}")
        logger.info(f"expected\t:\t{expected}")
        logger.info("---------")

        assert phonemes == expected


def test_g2p_as_binary():
    for index, (pron, accent, expected) in enumerate(
        [
            # For Basic Pattern
            ("ソレデワ", 3, "s00 o01 r00 e01 d00 e11 w00 a01"),
            ("カジノオ", 1, "k00 a11 j00 i01 n00 o01 o01"),
            # For ッ
            ("オクッテイル", 0, "o01 k00 u00 cl01 t00 e01 i01 r00 u01"),
            ("ハジマッタ", 0, "h00 a01 j00 i01 m00 a00 cl01 t00 a01"),
            ("ギインリッポー", 4, "g00 i01 i00 N01 r00 i10 cl01 p00 oo01"),
            # For long Vowel(ー)
            ("シンデイル", 0, "sh00 i00 N01 d00 e01 i01 r00 u01"),
            ("ユキオンナオ", 3, "y00 u01 k00 i01 o10 N01 n00 a01 o01"),
            ("ツーワリョーガ", 3, "ts00 uu01 w00 a11 ry00 oo01 g00 a01"),
            ("アソンデイル", 0, "a01 s00 o00 N01 d00 e01 i01 r00 u01"),
            ("トーナンアジアオ", 5, "t00 oo01 n00 a00 N01 a11 j00 i01 a01 o01"),
            ("サイセーシマス", 6, "s00 a01 i01 s00 ee01 sh00 i01 m00 a11 s00 u01"),
            (
                "トーゴーガタリゾートノ",
                8,
                "t00 oo01 g00 oo01 g00 a01 t00 a01 r00 i01 z00 oo11 t00 o01 n00 o01",
            ),
            # For ン(N)
            ("ンダモシタン", 4, "N01 d00 a01 m00 o01 sh00 i11 t00 a00 N01"),
            ("チョーセンスル", 0, "ch00 oo01 s00 e00 N01 s00 u01 r00 u01"),
            ("ヘンコーオ", 0, "h00 e00 N01 k00 oo01 o01"),
            # For long vowel + ン(N)
            ("バチーンッテ", 2, "b00 a01 ch00 ii10 N01 cl01 t00 e01"),
            ("バーンアウトワ", 4, "b00 aa00 N01 a11 u01 t00 o01 w00 a01"),
            # For ンー(NN)
            ("ンー", 1, "NN11"),
            ("ンー", 0, "NN01"),
            ("ジャンケンー", 1, "j00 a10 N01 k00 e00 NN01"),
            ("オニーチャンー", 2, "o01 n00 ii11 ch00 a00 NN01"),
            # For ッー(cl)
            ("ッー", 1, "cl11"),
            ("ッー", 0, "cl01"),
        ]
    ):
        phons, accents, boundaries = pron2phon(pron, accent, represent_mode="binary")

        phonemes = " ".join(
            [f"{p}{a}{b}" for p, a, b in zip(phons, accents, boundaries)]
        )

        logger.info(f"No.{index} {pron} / {accent}")
        logger.info(f"answer\t:\t{phonemes}")
        logger.info(f"expected\t:\t{expected}")
        logger.info("---------")

        assert phonemes == expected


def test_mora_split():
    for index, (pron, expected) in enumerate(
        [
            # For Basic Pattern
            ("ソレデワ", ["ソ", "レ", "デ", "ワ"]),
            ("カジノオ", ["カ", "ジ", "ノ", "オ"]),
            # For long Vowel(ー)
            ("シンデイル", ["シ", "ン", "デ", "イ", "ル"]),
            ("ユキオンナオ", ["ユ", "キ", "オ", "ン", "ナ", "オ"]),
            ("ツーワリョーガ", ["ツ", "ー", "ワ", "リョ", "ー", "ガ"]),
            ("ツーヮワリョーガ", ["ツ", "ー", "ヮ", "ワ", "リョ", "ー", "ガ"]),
            ("アソンデイル", ["ア", "ソ", "ン", "デ", "イ", "ル"]),
            ("トーナンアジアオ", ["ト", "ー", "ナ", "ン", "ア", "ジ", "ア", "オ"]),
            ("サイセーシマス", ["サ", "イ", "セ", "ー", "シ", "マ", "ス"]),
            ("トーゴーガタリゾートノ", ["ト", "ー", "ゴ", "ー", "ガ", "タ", "リ", "ゾ", "ー", "ト", "ノ"]),
            # For ン(N)
            ("ンダモシタン", ["ン", "ダ", "モ", "シ", "タ", "ン"]),
            ("チョーセンスル", ["チョ", "ー", "セ", "ン", "ス", "ル"]),
            ("ヘンコーオ", ["ヘ", "ン", "コ", "ー", "オ"]),
            ("ヘンコーオォ", ["ヘ", "ン", "コ", "ー", "オ", "ォ"]),
            # For long vowel + ン(N)
            ("バチーンッテ", ["バ", "チ", "ー", "ン", "ッ", "テ"]),
            ("バーンアウトワ", ["バ", "ー", "ン", "ア", "ウ", "ト", "ワ"]),
            # For ンー(NN)
            ("ンー", ["ン", "ー"]),
            ("ジャンケンー", ["ジャ", "ン", "ケ", "ン", "ー"]),
            ("オニーチャンー", ["オ", "ニ", "ー", "チャ", "ン", "ー"]),
            # For ッー(cl)
            ("ッー", ["ッ", "ー"]),
            ("ッョ", ["ッ", "ョ"]),
            ("オクッテイル", ["オ", "ク", "ッ", "テ", "イ", "ル"]),
            ("ハジマッタ", ["ハ", "ジ", "マ", "ッ", "タ"]),
            ("ギインリッポー", ["ギ", "イ", "ン", "リ", "ッ", "ポ", "ー"]),
        ]
    ):

        moras = pron2mora(pron)

        logger.info(f"No.{index} {pron}")
        logger.info(f"answer\t:\t{moras}")
        logger.info(f"expected\t:\t{expected}")
        logger.info("---------")

        assert moras == expected
