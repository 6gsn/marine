# Copyright 2019 Allen Institute for AI
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import random
from logging import getLogger
from typing import Any

import torch
from marine.modules.crf_tagger import logsumexp, viterbi_decode
from numpy.testing import assert_almost_equal, assert_equal

logger = getLogger("test")


def test_logsumexp():
    # First a simple example where we add probabilities in log space.
    tensor = torch.FloatTensor([[0.4, 0.1, 0.2]])
    log_tensor = tensor.log()
    log_summed = logsumexp(log_tensor, dim=-1, keepdim=False)
    assert_almost_equal(log_summed.exp().data.numpy(), [0.7])
    log_summed = logsumexp(log_tensor, dim=-1, keepdim=True)
    assert_almost_equal(log_summed.exp().data.numpy(), [[0.7]])

    # Then some more atypical examples, and making sure this will work with how we handle
    # log masks.
    tensor = torch.FloatTensor([[float("-inf"), 20.0]])
    assert_almost_equal(logsumexp(tensor).data.numpy(), [20.0])
    tensor = torch.FloatTensor([[-200.0, 20.0]])
    assert_almost_equal(logsumexp(tensor).data.numpy(), [20.0])
    tensor = torch.FloatTensor([[20.0, 20.0], [-200.0, 200.0]])
    assert_almost_equal(logsumexp(tensor, dim=0).data.numpy(), [20.0, 200.0])


def test_viterbi_decode():
    # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
    sequence_logits = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
    transition_matrix = torch.zeros([9, 9])
    indices, _ = viterbi_decode(sequence_logits.data, transition_matrix)
    _, argmax_indices = torch.max(sequence_logits, 1)
    assert indices == argmax_indices.data.squeeze().tolist()

    # Test Viterbi decoding works with start and end transitions
    sequence_logits = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
    transition_matrix = torch.zeros([9, 9])
    allowed_start_transitions = torch.zeros([9])
    # Force start tag to be an 8
    allowed_start_transitions[:8] = float("-inf")
    allowed_end_transitions = torch.zeros([9])
    # Force end tag to be a 0
    allowed_end_transitions[1:] = float("-inf")
    indices, _ = viterbi_decode(
        sequence_logits.data,
        transition_matrix,
        allowed_end_transitions=allowed_end_transitions,
        allowed_start_transitions=allowed_start_transitions,
    )
    assert indices[0] == 8
    assert indices[-1] == 0

    # Test that pairwise potentials affect the sequence correctly and that
    # viterbi_decode can handle -inf values.
    sequence_logits = torch.FloatTensor(
        [
            [0, 0, 0, 3, 5],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
        ]
    )
    # The same tags shouldn't appear sequentially.
    transition_matrix = torch.zeros([5, 5])
    for i in range(5):
        transition_matrix[i, i] = float("-inf")
    indices, _ = viterbi_decode(sequence_logits, transition_matrix)
    assert indices == [4, 3, 4, 3, 4, 3]

    # Test that unbalanced pairwise potentials break ties
    # between paths with equal unary potentials.
    sequence_logits = torch.FloatTensor(
        [
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
        ]
    )
    # The 5th tag has a penalty for appearing sequentially
    # or for transitioning to the 4th tag, making the best
    # path uniquely to take the 4th tag only.
    transition_matrix = torch.zeros([5, 5])
    transition_matrix[4, 4] = -10
    transition_matrix[4, 3] = -10
    transition_matrix[3, 4] = -10
    indices, _ = viterbi_decode(sequence_logits, transition_matrix)
    assert indices == [3, 3, 3, 3, 3, 3]

    sequence_logits = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
    # Best path would normally be [3, 2, 3] but we add a
    # potential from 2 -> 1, making [3, 2, 1] the best path.
    transition_matrix = torch.zeros([4, 4])
    transition_matrix[0, 0] = 1
    transition_matrix[2, 1] = 5
    indices, value = viterbi_decode(sequence_logits, transition_matrix)
    assert indices == [3, 2, 1]
    assert value.numpy() == 18

    # Test that providing evidence results in paths containing specified tags.
    sequence_logits = torch.FloatTensor(
        [
            [0, 0, 0, 7, 7],
            [0, 0, 0, 7, 7],
            [0, 0, 0, 7, 7],
            [0, 0, 0, 7, 7],
            [0, 0, 0, 7, 7],
            [0, 0, 0, 7, 7],
        ]
    )
    # The 5th tag has a penalty for appearing sequentially
    # or for transitioning to the 4th tag, making the best
    # path to take the 4th tag for every label.
    transition_matrix = torch.zeros([5, 5])
    transition_matrix[4, 4] = -10
    transition_matrix[4, 3] = -2
    transition_matrix[3, 4] = -2
    # The 1st, 4th and 5th sequence elements are observed - they should be
    # equal to 2, 0 and 4. The last tag should be equal to 3, because although
    # the penalty for transitioning to the 4th tag is -2, the unary potential
    # is 7, which is greater than the combination for any of the other labels.
    observations = [2, -1, -1, 0, 4, -1]
    indices, _ = viterbi_decode(sequence_logits, transition_matrix, observations)
    assert indices == [2, 3, 3, 0, 4, 3]


def test_viterbi_decode_top_k():
    # Test cases taken from: https://gist.github.com/PetrochukM/afaa3613a99a8e7213d2efdd02ae4762

    # Test Viterbi decoding is equal to greedy decoding with no pairwise potentials.
    sequence_logits = torch.autograd.Variable(torch.rand([5, 9]))
    transition_matrix = torch.zeros([9, 9])

    indices, _ = viterbi_decode(sequence_logits.data, transition_matrix, top_k=5)

    _, argmax_indices = torch.max(sequence_logits, 1)
    assert indices[0] == argmax_indices.data.squeeze().tolist()

    # Test that pairwise potentials effect the sequence correctly and that
    # viterbi_decode can handle -inf values.
    sequence_logits = torch.FloatTensor(
        [
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 3, 4],
        ]
    )
    # The same tags shouldn't appear sequentially.
    transition_matrix = torch.zeros([5, 5])
    for i in range(5):
        transition_matrix[i, i] = float("-inf")
    indices, _ = viterbi_decode(sequence_logits, transition_matrix, top_k=5)
    assert indices[0] == [3, 4, 3, 4, 3, 4]

    # Test that unbalanced pairwise potentials break ties
    # between paths with equal unary potentials.
    sequence_logits = torch.FloatTensor(
        [
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 0],
        ]
    )
    # The 5th tag has a penalty for appearing sequentially
    # or for transitioning to the 4th tag, making the best
    # path uniquely to take the 4th tag only.
    transition_matrix = torch.zeros([5, 5])
    transition_matrix[4, 4] = -10
    transition_matrix[4, 3] = -10
    indices, _ = viterbi_decode(sequence_logits, transition_matrix, top_k=5)
    assert indices[0] == [3, 3, 3, 3, 3, 3]

    sequence_logits = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
    # Best path would normally be [3, 2, 3] but we add a
    # potential from 2 -> 1, making [3, 2, 1] the best path.
    transition_matrix = torch.zeros([4, 4])
    transition_matrix[0, 0] = 1
    transition_matrix[2, 1] = 5
    indices, value = viterbi_decode(sequence_logits, transition_matrix, top_k=5)
    assert indices[0] == [3, 2, 1]
    assert value[0] == 18

    def _brute_decode(
        tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int = 5
    ) -> Any:
        """
        Top-k decoder that uses brute search
        instead of the Viterbi Decode dynamic programing algorithm
        """
        # Create all possible sequences
        sequences = [[]]  # type: ignore

        for i in range(len(tag_sequence)):
            new_sequences = []  # type: ignore
            for j in range(len(tag_sequence[i])):
                for sequence in sequences:
                    new_sequences.append(sequence[:] + [j])
            sequences = new_sequences

        # Score
        scored_sequences = []  # type: ignore
        for sequence in sequences:
            emission_score = sum(tag_sequence[i, j] for i, j in enumerate(sequence))
            transition_score = sum(
                transition_matrix[sequence[i - 1], sequence[i]]
                for i in range(1, len(sequence))
            )
            score = emission_score + transition_score
            scored_sequences.append((score, sequence))

        # Get the top k scores / paths
        top_k_sequences = sorted(scored_sequences, key=lambda r: r[0], reverse=True)[
            :top_k
        ]
        scores, paths = zip(*top_k_sequences)

        return paths, scores  # type: ignore

    def _sanitize(x: Any) -> Any:
        """
        Sanitize turns PyTorch and Numpy types into basic Python types
        """
        if isinstance(x, (str, float, int, bool)):
            return x
        elif isinstance(x, torch.Tensor):
            return x.cpu().tolist()
        elif isinstance(x, (list, tuple, set)):
            return [_sanitize(x_i) for x_i in x]
        else:
            raise ValueError(f"Cannot sanitize {x} of type {type(x)}. ")

    # Run 100 randomly generated parameters and compare the outputs.
    for _ in range(100):
        num_tags = random.randint(1, 5)
        seq_len = random.randint(1, 5)
        k = random.randint(1, 5)
        sequence_logits = torch.rand([seq_len, num_tags])
        transition_matrix = torch.rand([num_tags, num_tags])
        viterbi_paths_v1, viterbi_scores_v1 = viterbi_decode(
            sequence_logits, transition_matrix, top_k=k
        )
        viterbi_path_brute, viterbi_score_brute = _brute_decode(
            sequence_logits, transition_matrix, top_k=k
        )
        assert_almost_equal(
            list(viterbi_score_brute), viterbi_scores_v1.tolist(), decimal=3
        )
        assert_equal(_sanitize(viterbi_paths_v1), viterbi_path_brute)
