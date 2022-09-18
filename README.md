# **MARINE** : **M**ulti-task lea**R**n**I**ng-based Japa**N**ese accent **E**stimation

[![PyPI](https://img.shields.io/pypi/v/marine.svg)](https://pypi.python.org/pypi/marine)
[![Python package](https://github.com/6gsn/marine/actions/workflows/ci.yml/badge.svg)](https://github.com/6gsn/marine/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.md)
<!-- [![DOI](https://zenodo.org/badge/#)](https://zenodo.org/badge/latestdoi/#) -->

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

## Get started

### Install

```shell
$ pip install marine
```

### For develop

```shell
$ pip install -e ".[dev]"
```

### Quick demo

```python
from marine.predict import Predictor

```

## LICENSE

- marine: Apache_2.0 license ([LICENSE.md](LICENSE.md))
- JSUT: Modified BSD license ([COPYING](https://github.com/r9y9/open_jtalk/blob/1.10/src/COPYING))