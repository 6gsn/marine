# coding: utf-8

import re

_re_has_longvowel = None
_re_hiragana = None
_re_katakana = None
_re_kanji = None
_re_letter = None
_re_symbol = None


def is_hiragana(surface):
    global _re_hiragana
    if _re_hiragana is None:
        _re_hiragana = re.compile(r"^[ぁ-ん]+$")
    return _re_hiragana.match(surface) is not None


def is_katakana(surface):
    global _re_katakana
    if _re_katakana is None:
        _re_katakana = re.compile(r"^[ァ-ヴヶ]+$")
    return _re_katakana.match(surface) is not None


def is_kanji(surface):
    global _re_kanji
    if _re_kanji is None:
        _re_kanji = re.compile(r"^[一-龠]+$")
    return _re_kanji.match(surface) is not None


def is_letter(surface):
    global _re_letter
    if _re_letter is None:
        _re_letter = re.compile(r"^[a-zA-Z]+$")
    return _re_letter.match(surface) is not None


def is_symbol(surface):
    global _re_symbol
    if _re_symbol is None:
        _re_symbol = re.compile(r"^[〆々ー,.?!]+$")
    return _re_symbol.match(surface) is not None


def has_longvowel(text):
    global _re_has_longvowel
    if _re_has_longvowel is None:
        _re_has_longvowel = re.compile(r"(aa|ii|uu|ee|oo)$")
    return _re_has_longvowel.search(text) is not None
