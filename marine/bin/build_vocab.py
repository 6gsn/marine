import argparse
import sys
from collections import defaultdict
from pathlib import Path

from joblib import dump
from marine.logger import getLogger
from marine.utils.util import load_json_corpus
from tqdm import tqdm

logger = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate vocabulary file for word-embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("in_path", type=Path, help="Path or directory for feature file")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--min_freq",
        "-m",
        type=int,
        default=2,
        help="""
        Minimum word frequency in the whole corpus,
        which to judge whether the word include in the vocabulary
        """,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Whether print log for debug",
    )
    return parser


def count_words(words):
    freqs = defaultdict(int)

    for word in tqdm(words, "Counting word", leave=False):
        freqs[word] += 1

    freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

    return freqs


def filter_words(freqs, min_freq):
    return list(filter(lambda x: x[1] >= min_freq, freqs))


def save_vocab(freqs, output_dir):
    words = [surface for surface, _ in freqs]
    dump(words, output_dir / "vocab.pkl", compress=True)


def entry(argv=sys.argv):
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(f"Loaded parameters: {args}")

    input_path = args.in_path
    output_dir = args.out_dir

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    corpus = load_json_corpus(input_path)
    words = [node["surface"] for script in corpus for node in script["nodes"]]
    logger.info(f"Loaded {len(words):,} words")

    freqs = count_words(words)
    logger.info(f"Get {len(freqs):,} unique words")

    freqs = filter_words(freqs, args.min_freq)
    logger.info(f"Filtered {len(freqs):,} words")

    save_vocab(freqs, output_dir)


if __name__ == "__main__":
    sys.exit(entry())
