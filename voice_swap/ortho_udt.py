"""
Orthogonal unsupervised domain translation, as implemented by
https://arxiv.org/abs/2007.12568.
"""

import numpy as np

from .neighbors import Neighbors


def ortho_udt(
    source, target, verbose=False, no_cycle_check=False, max_iters=None, orthogonal=True
):
    """
    Compute an orthogonal matrix that translates from a source domain to a
    target domain.

    :param source: an [N x D] array of source vectors.
    :param target: an [N x D] array of target vectors.
    :param verbose: if True, log information during optimization.
    :param no_cycle_check: if True, don't enforce cycle consistency.
    :param max_iters: if specified, the maximum iteration count.
    :param orthogonal: if False, create an arbitrary matrix instead of an
                       orthogonal one.
    :return: a [D x D] orthogonal matrix that takes vectors from the source
             and produces vectors from the target.
    """
    neighbor_calc = Neighbors.create(verbose=verbose)
    solution = np.eye(source.shape[1])
    num_iters = 0
    while True:
        new_source = source @ solution
        if verbose:
            print("Computing neighbors...")
        source_neighbors, target_neighbors = neighbor_calc.neighbors(new_source, target)

        best_buddies = target_neighbors[source_neighbors] == np.arange(len(new_source))

        stats = {
            "best_buddies": np.sum(best_buddies),
            "uniq_targets": len(set(source_neighbors)),
            "uniq_sources": len(set(target_neighbors)),
        }

        all_source_vecs = np.concatenate([source, target], axis=0)
        all_target_vecs = np.concatenate(
            [target[source_neighbors], source[target_neighbors]], axis=0
        )
        stats["all_mse"] = np.mean((all_source_vecs - all_target_vecs) ** 2)

        bb_source_vecs = source[best_buddies]
        bb_target_vecs = target[source_neighbors[best_buddies]]
        stats["bb_mse"] = np.mean((bb_source_vecs - bb_target_vecs) ** 2)

        if no_cycle_check:
            source_vecs, target_vecs = all_source_vecs, all_target_vecs
        else:
            source_vecs, target_vecs = bb_source_vecs, bb_target_vecs

        if orthogonal:
            u, _, vh = np.linalg.svd(source_vecs.T @ target_vecs)
            new_solution = u @ vh
        else:
            new_solution = (
                np.linalg.inv(source_vecs.T @ source_vecs) @ source_vecs.T @ target_vecs
            )

        stats["identity_dist"] = np.mean(
            (np.eye(len(new_solution)) - new_solution) ** 2
        )

        num_iters += 1

        if verbose:
            stats["iter"] = num_iters
            display_dict(stats)

        if np.allclose(solution, new_solution) or num_iters == max_iters:
            break

        solution = new_solution
    return solution


def display_dict(d):
    key_length = max(len(x) for x in d.keys())
    value_length = max(len(f"{x}") for x in d.values())
    print("-" * (key_length + value_length + 7))
    for key, value in d.items():
        pad_key = pad_spaces(key, key_length)
        pad_value = pad_spaces(value, value_length)
        print(f"| {pad_key} | {pad_value} |")
    print("-" * (key_length + value_length + 7))


def pad_spaces(value, length):
    res = f"{value}"
    while len(res) < length:
        res += " "
    return res
