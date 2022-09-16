import pytest
from marine.predict import Predictor


@pytest.fixture
def predictor() -> Predictor:
    """load inference model using default config"""
    return Predictor()


def test_predict(predictor):
    """just to confirm predict() is working without errors."""
    nodes = [
        {
            "surface": "水",
            "pron": "ミズ",
            "pos": "名詞:一般:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "C3",
            "chain_flag": -1,
        },
        {
            "surface": "を",
            "pron": "オ",
            "pos": "助詞:格助詞:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "動詞%F5,名詞%F1",
            "chain_flag": 1,
        },
        {
            "surface": "マレーシア",
            "pron": "マレーシア",
            "pos": "名詞:固有名詞:地域:国",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "C1",
            "chain_flag": 0,
        },
        {
            "surface": "から",
            "pron": "カラ",
            "pos": "助詞:格助詞:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "名詞%F1",
            "chain_flag": 1,
        },
        {
            "surface": "買わ",
            "pron": "カワ",
            "pos": "動詞:自立:*:*",
            "c_type": "五段・ワ行促音便",
            "c_form": "未然形",
            "accent_type": 0,
            "accent_con_type": "*",
            "chain_flag": 0,
        },
        {
            "surface": "なく",
            "pron": "ナク",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・ナイ",
            "c_form": "連用テ接続",
            "accent_type": 1,
            "accent_con_type": "動詞%F3@0",
            "chain_flag": 1,
        },
        {
            "surface": "て",
            "pron": "テ",
            "pos": "助詞:接続助詞:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "動詞%F1,形容詞%F1,名詞%F5",
            "chain_flag": 1,
        },
        {
            "surface": "は",
            "pron": "ワ",
            "pos": "助詞:係助詞:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "名詞%F1,動詞%F2@0,形容詞%F2@0",
            "chain_flag": 1,
        },
        {
            "surface": "なら",
            "pron": "ナラ",
            "pos": "動詞:非自立:*:*",
            "c_type": "五段・ラ行",
            "c_form": "未然形",
            "accent_type": 2,
            "accent_con_type": "*",
            "chain_flag": 0,
        },
        {
            "surface": "ない",
            "pron": "ナイ",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・ナイ",
            "c_form": "基本形",
            "accent_type": 1,
            "accent_con_type": "動詞%F3@0,形容詞%F2@1",
            "chain_flag": 1,
        },
        {
            "surface": "の",
            "pron": "ノ",
            "pos": "名詞:非自立:一般:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 2,
            "accent_con_type": "動詞%F2@0,形容詞%F2@-1",
            "chain_flag": 0,
        },
        {
            "surface": "です",
            "pron": "デス",
            "pos": "助動詞:*:*:*",
            "c_type": "特殊・デス",
            "c_form": "基本形",
            "accent_type": 1,
            "accent_con_type": "名詞%F2@1,動詞%F1,形容詞%F2@0",
            "chain_flag": 1,
        },
        {
            "surface": ".",
            "pron": None,
            "pos": "記号:句点:*:*",
            "c_type": "*",
            "c_form": "*",
            "accent_type": 0,
            "accent_con_type": "*",
            "chain_flag": 0,
        }
    ]
    feature = predictor.predict([nodes])

    print(feature)
