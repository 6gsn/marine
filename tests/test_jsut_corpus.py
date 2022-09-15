from logging import getLogger

from marine.bin.jsut2corpus import parse_jsut_annotation

logger = getLogger("test")


def test_jsut_parser():
    for (
        jsut_annotaion,
        accent_status_seq_level,
        accent_status_represent_mode,
        expect,
    ) in [
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "ap",
            "binary",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,5,4,0",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "mora",
            "binary",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
        (
            "^ム]シロ#ロ[ンゲノホ]ーガ#ハ[ゲヤス]イッテ#キ[ータゾ?$",
            "mora",
            "high_low",
            {
                "pron": "ムシロロンゲノホーガハゲヤスイッテキータゾ",
                "accent_status": "1,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1",
                "accent_phrase_boundary": "0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0",
                "intonation_phrase_boundary": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            },
        ),
    ]:
        result = parse_jsut_annotation(
            jsut_annotaion, accent_status_seq_level, accent_status_represent_mode
        )
        assert result == expect
