# **MARINE** : **M**ulti-task lea**R**n**I**ng-based Japa**N**ese accent **E**stimation

[![PyPI](https://img.shields.io/pypi/v/marine.svg)](https://pypi.python.org/pypi/marine)
[![Python package](https://github.com/6gsn/marine/actions/workflows/ci.yml/badge.svg)](https://github.com/6gsn/marine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7092054.svg)](https://doi.org/10.5281/zenodo.7092054)

`marine` is a tool kit for building the Japanese accent estimation model proposed in [our paper](https://www.isca-speech.org/archive/interspeech_2022/park22b_interspeech.html).

For academic use, please cite the following paper ([IEEE Xplore](https://www.isca-speech.org/archive/interspeech_2022/park22b_interspeech.html)).

```bibtex
@inproceedings{park22b_interspeech,
  author={Byeongseon Park and Ryuichi Yamamoto and Kentaro Tachibana},
  title={{A Unified Accent Estimation Method Based on Multi-Task Learning for Japanese Text-to-Speech}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={1931--1935},
  doi={10.21437/Interspeech.2022-334}
}
```

## Notice

The model included in this package is trained using [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut), which is not the same as the dataset in [our paper](https://www.isca-speech.org/archive/interspeech_2022/park22b_interspeech.html). Therefore, the model's performance is also not equal to the performance introduced in our paper.

## Get started

### Installation

```shell
$ pip install marine
```

### For development

```shell
$ pip install -e ".[dev]"
```

### Quick demo

```python
In [1]: from marine.predict import Predictor

In [2]: nodes = [{"surface": "こんにちは", "pos": "感動詞:*:*:*", "pron": "コンニチワ", "c_type": "*", "c_form": "*", "accent_type": 0, "accent_con_type": "-1", "chain_flag": -1}]

In [3]: predictor = Predictor()

In [4]: predictor.predict([nodes])
Out[4]:
{'mora': [['コ', 'ン', 'ニ', 'チ', 'ワ']],
 'intonation_phrase_boundary': [[0, 0, 0, 0, 0]],
 'accent_phrase_boundary': [[0, 0, 0, 0, 0]],
 'accent_status': [[0, 0, 0, 0, 0]]}

In [5]: predictor.predict([nodes], accent_represent_mode="high_low")
Out[5]:
{'mora': [['コ', 'ン', 'ニ', 'チ', 'ワ']],
 'intonation_phrase_boundary': [[0, 0, 0, 0, 0]],
 'accent_phrase_boundary': [[0, 0, 0, 0, 0]],
 'accent_status': [[0, 1, 1, 1, 1]]}
```

### Build model yourself

Coming soon...

## LICENSE

- marine: Apache 2.0 license ([LICENSE](LICENSE))
- JSUT: CC-BY-SA 4.0 license, etc. (Please check [jsut-label/LICENCE.txt](https://github.com/sarulab-speech/jsut-label/blob/master/LICENCE.txt))