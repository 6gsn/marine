# flake8: noqa: B023

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from joblib import dump, load
from marine.data.feature.feature_set import FeatureSet
from marine.logger import getLogger
from marine.utils.g2p_util import pron2mora
from marine.utils.util import load_json_corpus, split_corpus
from tqdm import tqdm

logger = None

AP_BOUNDARY_LABEL = 1


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Special format txt format data to json file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("corpus_path", type=Path, help="Path or directory for corpus")
    parser.add_argument(
        "feature_path", type=Path, help="Path or directory for feature file"
    )
    parser.add_argument("vocab_path", type=Path, help="Vocab file path")
    parser.add_argument("out_dir", type=Path, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument(
        "--feature_table_key",
        "-f",
        type=str,
        choices=["unidic-csj", "open-jtalk"],
        default="unidic-csj",
        help="Sequence level for accent status label",
    )
    parser.add_argument(
        "--max_size",
        "-m",
        type=int,
        default=-1,
        help="""Maximum number of scripts to convert feature
        (-m < 0 = use all scripts)""",
    )
    parser.add_argument(
        "--accent_status_seq_level",
        "-s",
        type=str,
        choices=["ap", "mora"],
        default="ap",
        help="Sequence level for accent status label",
    )
    parser.add_argument(
        "--test_size",
        "-t",
        type=int,
        default=-1,
        help="""Specific size of samples for val and test
        (-t < 0 = 5%% and / 5%% for val and test respectively)""",
    )
    parser.add_argument(
        "--skip_corpus_split",
        action="store_true",
        help="Whether skip corpus splitting",
    )
    parser.add_argument(
        "--single_corpus_key",
        "-k",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="""Dataset key to export
        (this option will be ignored if --skip_corpus_split is not activated)""",
    )
    parser.add_argument(
        "--target_id_dir",
        type=Path,
        default=None,
        help="Directory of id files to reproduce specific dataset",
    )
    parser.add_argument(
        "--random_seed",
        "-r",
        type=int,
        default=12345,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=50,
        help="Logging level",
    )
    return parser


def insert_punctuation_by_extracted_features(
    mora_seq_w_puncts,
    mora_seq_wo_puncts,
    labels,
    punctuation_ids,
    accent_status_seq_level,
):
    indexs = np.where(np.in1d(mora_seq_w_puncts, punctuation_ids))[0]

    for index in indexs:
        mora_seq_wo_puncts = np.insert(
            mora_seq_wo_puncts, index, int(mora_seq_w_puncts[index])
        )
        for label_key in labels.keys():
            if label_key == "accent_status" and accent_status_seq_level == "ap":
                # the AP-based AN label represents the position of AN in a AP
                # i.e., Don't need consider the global indexå
                # Also, the accent nucleus never located at punctuation
                continue

            # the label where punctuation are must be 0
            labels[label_key].insert(index, 0)
            assert len(mora_seq_wo_puncts) == len(
                labels[label_key]
            ), f"Length of inserted result is not matched: {len(mora_seq_wo_puncts)} != {len(labels[label_key])} ({label_key})"

    assert np.array_equal(
        mora_seq_w_puncts, mora_seq_wo_puncts
    ), f"The inserted mora seq not same with original mora seq: {mora_seq_w_puncts} != {mora_seq_wo_puncts}"

    return labels


def _process(
    nodes,
    feature_set,
    accent_status_seq_level,
    script_id,
    surface,
    pron,
    **original_labels,
):
    if len(nodes) == 0:
        if logger is not None:
            logger.debug(f"Empty nodes: {script_id}")
        return None

    # get punctuation id
    punctuation_ids = feature_set.get_punctuation_ids()

    # check accent seq level
    required_ap_accent = (
        "accent" in original_labels.keys() and accent_status_seq_level == "ap"
    )

    # init labels, features
    labels = {key: [] for key in original_labels.keys()}
    _original_labels = {
        key: [int(v) for v in value.split(",")]
        for key, value in original_labels.items()
    }

    # convert nodes to feature seqs
    feature = feature_set.convert_nodes_to_feature(nodes)

    # convert original pron in annotation to id seqs
    original_mora = pron2mora(pron)
    expected_ids = feature_set.convert_feature_to_id("mora", original_mora)

    # verify the features is available
    # i.e., is the prounnounces is same w/o punctuation
    punct_removed_extracted_mora = feature["mora"][
        np.in1d(feature["mora"], punctuation_ids, invert=True)
    ]
    punct_removed_expected_mora = expected_ids[
        np.in1d(expected_ids, punctuation_ids, invert=True)
    ]

    if np.array_equal(punct_removed_extracted_mora, punct_removed_expected_mora):
        if len(feature["mora"]) != len(expected_ids):
            _original_labels = insert_punctuation_by_extracted_features(
                feature["mora"],
                expected_ids,
                _original_labels,
                punctuation_ids,
                accent_status_seq_level,
            )

        for key in labels.keys():
            # verify for ap-based AN labels with AP labels
            if (
                required_ap_accent
                and key == "accent_status"
                and "accent_phrase_boundary" in labels.keys()
            ):
                num_boundary = (
                    np.count_nonzero(
                        _original_labels["accent_phrase_boundary"] == AP_BOUNDARY_LABEL
                    )
                    + 1
                )
                accents = _original_labels["accent_status"]
                assert (
                    len(accents) == num_boundary
                ), "Unmatched length of sequnce between ac and ap"
                labels["accent_status"] = np.array(
                    _original_labels["accent_status"], dtype=np.uint8
                )
            else:
                labels[key] = np.array(_original_labels[key], dtype=np.uint8)
    else:
        if logger is not None:
            logger.debug(
                (
                    f"Wrong mora [{script_id}]:"
                    f" {''.join(feature_set.convert_id_to_feature('mora', punct_removed_expected_mora))}"
                    f" != {''.join(feature_set.convert_id_to_feature('mora', punct_removed_extracted_mora))}"
                )
            )
        return None

    return script_id, feature, labels


def _load_target_ids(target_id_dir):
    id_groups = {}

    logger.info(f"Load ids from existing dataset ({target_id_dir})")
    for path in target_id_dir.glob("*/ids.pkl"):
        dataset_key = str(path.parent.name)
        id_groups[dataset_key] = set(load(path))
        logger.info(
            f"Loaded {len(id_groups[dataset_key]):,} ids for {dataset_key} in {target_id_dir}"
        )

    return id_groups


def _sort_corpus_by_script_id(corpus):
    return list(sorted(corpus, key=lambda x: x[0]))


def _remove_unavailable_script(corpus):
    return list(filter(lambda x: x is not None, corpus))


def _split_corpus_by_ids(corpus, id_groups):
    _corpus = {}

    for key, target_ids in id_groups.items():
        scripts = list(filter(lambda x: x[0] in target_ids, corpus))
        assert len(scripts) == len(
            target_ids
        ), f"Not enough number of scripts for {key}: {len(scripts):,} != {len(target_ids):,}"
        _corpus[key] = scripts

    # When specified the ID group including in val, test
    if set(["val", "test"]) == set(_corpus.keys()):
        _corpus["train"] = list(
            filter(
                lambda x: x[0] not in id_groups["val"]
                and x[0] not in id_groups["test"],
                corpus,
            )
        )
    # When the ID group is specified fully (i.e., train, val and test)
    elif set(["train", "val", "test"]) == set(_corpus.keys()):
        pass
    else:
        raise NotImplementedError(
            f"Not supported ID group specification: {_corpus.keys()}"
        )

    return _corpus


def entry(argv=sys.argv):
    global logger

    args = get_parser().parse_args(argv[1:])
    logger = getLogger(args.verbose)
    logger.debug(args)

    # Init feature-set
    feature_set = FeatureSet(args.vocab_path, feature_table_key=args.feature_table_key)

    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    # Load corpus
    corpus = load_json_corpus(args.corpus_path)
    features = load_json_corpus(args.feature_path)

    assert len(corpus) == len(
        features
    ), "Not match script size between corpus and feature files"
    assert [script["script_id"] for script in corpus] == [
        script["script_id"] for script in features
    ], "Not match script ids between corpus and feature files"

    if args.max_size > 0:
        assert (
            len(corpus) > args.max_size
        ), f"Not enough number of script: {len(corpus)} < {args.max_size}"
        corpus = corpus[: args.max_size]
        features = features[: args.max_size]

    # Process
    n_jobs = min(cpu_count(), args.n_jobs)

    if n_jobs > 1:
        logger.info(f"Processing {len(corpus):,} scripts with {n_jobs} jobs")
        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(
                    _process,
                    feature["nodes"],
                    feature_set,
                    args.accent_status_seq_level,
                    **script,
                )
                for script, feature in zip(corpus, features)
            ]
            corpus = [
                future.result()
                for future in tqdm(
                    futures, desc="Convert corpus to feature", leave=False
                )
            ]
    else:
        logger.info(f"Processing {len(corpus):,} scripts in a single thread")
        corpus = [
            _process(
                feature["nodes"], feature_set, args.accent_status_seq_level, **script
            )
            for script, feature in tqdm(
                zip(corpus, features), desc="Convert corpus to feature", leave=False
            )
        ]

    # Cleaing
    corpus = _sort_corpus_by_script_id(_remove_unavailable_script(corpus))
    logger.info(f"Cleaned corpus: {len(corpus):,}")

    if args.target_id_dir is None:
        if args.skip_corpus_split:
            corpus = {key: corpus for key in [args.single_corpus_key]}
        else:
            corpus = split_corpus(
                corpus, random_state=args.random_seed, absolute_test_size=args.test_size
            )
    else:
        id_groups = _load_target_ids(args.target_id_dir)
        corpus = _split_corpus_by_ids(corpus, id_groups)

    # Output results
    for key in corpus:
        phase_dir = args.out_dir / key
        phase_dir.mkdir(parents=True, exist_ok=True)

        _corpus = _sort_corpus_by_script_id(corpus[key])

        ids, features, labels = zip(*_corpus)
        logger.info(f"Saving {key} file with {len(labels):,} corpus in {args.out_dir}")

        dump(list(ids), phase_dir / "ids.pkl", compress=True)
        dump(list(features), phase_dir / "features.pkl", compress=True)
        dump(list(labels), phase_dir / "labels.pkl", compress=True)


if __name__ == "__main__":
    sys.exit(entry())
