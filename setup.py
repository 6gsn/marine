import codecs
import re
from os.path import exists, join

from setuptools import find_packages, setup


def find_version(*file_paths: str) -> str:
    with codecs.open(join(*file_paths), "r") as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if exists("README.md"):
    with open("README.md", "r") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""


setup(
    name="marine",
    version=find_version("marine", "__init__.py"),
    description="Marine: Multi-task learning based on Japanese accent estimation",
    packages=find_packages(),
    author="Byeongseon Park",
    author_email="6gsn.park@gmail.com",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "numpy >= 1.21.0",
        "torch >= 1.7.0",
        "hydra-core >= 1.1.0",
        "hydra_colorlog >= 1.1.0",
        "tqdm",
        "joblib",
    ],
    extras_require={
        "dev": [
            "torchmetrics",
            "scikit-learn",
            "docstr-coverage",
            "tensorboard",
            "matplotlib",
            "pytest",
            "pytest-cov",
            "docstr-coverage",
            "pysen",
            "black>=19.19b0,<=20.8",
            "flake8>=3.7,<4",
            "flake8-bugbear",
            "isort>=4.3,<5.2.0",
            "click<8.1.0",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx_rtd_theme",
            "nbsphinx>=0.8.6",
            "Jinja2>=3.0.1",
            "pandoc",
            "ipython",
            "jupyter",
        ],
        "pyopenjtalk": ["pyopenjtalk"],
    },
    entry_points={
        "console_scripts": [
            "marine-jsut2corpus = marine.bin.jsut2corpus:entry",
            "marine-build-vocab = marine.bin.build_vocab:entry",
            "marine-pack-corpus = marine.bin.pack_corpus:entry",
            "marine-train = marine.bin.train:entry",
            "marine-test = marine.bin.test:entry",
        ],
    },
)
