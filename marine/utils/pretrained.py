# Acknowledgement: some of the code was adapted from ttslearn
#  Copyright 2021 Ryuichi Yamamoto (MIT License)

import os
import shutil
import tarfile
from os.path import join
from pathlib import Path
from urllib.request import urlretrieve

from tqdm.auto import tqdm

DEFAULT_CACHE_DIR = join(os.path.expanduser("~"), ".cache", "marine")
CACHE_DIR = os.environ.get("MARINE_CACHE_DIR", DEFAULT_CACHE_DIR)

DEFAULT_VERSION = "v0.0.2"
MODEL_BASE_URL = "https://github.com/6gsn/marine/releases/download/"


# https://github.com/tqdm/tqdm#hooks-and-callbacks
class _TqdmUpTo(tqdm):  # type: ignore
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


def retrieve_pretrained_model(version=None):
    """Retrieve pretrained model from local cache or download from GitHub.
    Args:
        version (str): Version of pretrained model.
    Returns:
        str: Path to the pretrained model.
    Raises:
        ValueError: If the pretrained model is not found.
    Examples:
        >>> from marine.utils.pretrained import retrieve_pretrained_model
        >>> from marine.predict import Predictor
        >>> model_dir = retrieve_pretrained_model("v0.0.2")
        >>> predictor = Tacotron2PWGTTS(model_dir=model_dir, device="cpu")
    """

    if version is None:
        version = DEFAULT_VERSION
    elif not isinstance(version, str):
        raise TypeError(f"version must be str not {type(version)}")

    url = MODEL_BASE_URL + f"{version}/model.tar.gz"

    # NOTE: assuming that filename and extracted is the same
    out_dir = Path(CACHE_DIR) / version
    filename = Path(CACHE_DIR) / f"{version}/model.tar.gz"

    # re-download models
    if out_dir.exists() and len(list(out_dir.glob("*.pth"))) == 0:
        shutil.rmtree(out_dir)

    if not out_dir.exists():
        print(
            "The use of pre-trained models is permitted for non-commercial use only."
            "Please visit https://github.com/6gsn/marine to confirm the license."
        )
        print('Downloading: "{}"'.format(url))

        out_dir.mkdir(parents=True, exist_ok=True)

        with _TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=f"{version}/model.tar.gz",
        ) as t:  # all optional kwargs
            urlretrieve(url, filename, reporthook=t.update_to)
            t.total = t.n
        with tarfile.open(filename, mode="r|gz") as f:
            f.extractall(path=out_dir)
        os.remove(filename)

    return out_dir
