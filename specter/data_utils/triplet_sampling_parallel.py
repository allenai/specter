"""
This is identical to the original triplet_sampling.py code but optimized to run in parallel
The script reads the coview/co-citation data and returns them as triplet format
i.e., ['Query_paper', ('Positive_paper_id', num-coviews), ('Negative_paper_id', num_coviews)
"""
import functools
import multiprocessing
import operator
from typing import List, Tuple, Dict, Optional, Generator, NoReturn, Iterator

import math
import numpy as np
import logging
import tqdm
import random

from multiprocessing import Pool
import multiprocessing

logger = logging.getLogger(__file__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

# positive example: high coviews
# easy: those that have 0 coviews
# hard: those that have non-zero coviews but have smaller coviews than the positive
# note - hard will only be possible if there are at least 2 non-zero coview papers

np.random.seed(321)
random.seed(321)


def is_int(n):
    """ checks if a number is float. 2.0 is True while 0.3 is False"""
    return round(n) == n

_coviews = None
_margin_fraction = None
_paper_ids_set = None
_samples_per_query = None
_ratio_hard_negatives = None


# def _get_triplet(query, coviews, margin_fraction, paper_ids_set, samples_per_query, ratio_hard_negatives):
def _get_triplet(query):
    global _coviews
    global _margin_fraction
    global _paper_ids_set
    global _samples_per_query
    global _ratio_hard_negatives
    if query not in _coviews:
        return
    # self.coviews[query] is a dictionary of format {paper_id: {count: 1, frac: 1}}
    candidates = [(k, v['count']) for k, v in _coviews[query].items()]
    candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
    if len(candidates) > 1:
        coview_spread = candidates[0][1] - candidates[-1][1]
    else:
        coview_spread = 0
    margin = _margin_fraction * coview_spread  # minimum margin of coviews between good and bad sample

    # If distance is 1 increase margin to 1 otherwise any margin_fraction will pass
    if is_int(candidates[0][1]) and is_int(candidates[-1][1]) and coview_spread == 1:
        margin = np.ceil(margin)


    results = []

    # -------- hard triplets
    if len(candidates) > 2 and margin != 0:

        # find valid candidates by going through sorted
        # list and finding index of first sample with max coviews - margin
        for j in range(len(candidates)):
            if candidates[j][1] < (candidates[0][1] - margin):
                candidates_hard_neg = candidates[j:]
                break
            else:
                candidates_hard_neg = []

        neg_len = len(candidates_hard_neg)
        pos_len = len(candidates) - neg_len

        if neg_len > 0:

            # generate hard candidates
            n_hard_samples = math.ceil(_ratio_hard_negatives * _samples_per_query)
            # if there aren't enough candidates to generate enough unique samples
            # reduce the number of samples to make it possible for them to be unique
            if (pos_len * neg_len) < n_hard_samples:
                n_hard_samples = pos_len * neg_len

            for i in range(n_hard_samples):
                # find the negative sample first.
                neg = candidates_hard_neg[np.random.randint(len(candidates_hard_neg))]  # random neg sample from candidates

                candidates_pos = []
                # find the good sample. find valid candidates by going through sorted list
                # in reverse and finding index of first sample with bad sample + margin
                for j in range(len(candidates) - 1, -1, -1):
                    if candidates[j][1] > (neg[1] + margin):
                        candidates_pos = candidates[0:j + 1]
                        break

                if candidates_pos:
                    pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random pos sample from candidates

                    # append the good and bad samples with their coview number to output
                    results.append([query, pos, neg])

        n_easy_samples = _samples_per_query - len(results)

        # ---------- easy triplets

        # valid candidates are those with zeros
        candidates_zero = list(_paper_ids_set.difference([i[0] for i in candidates] + [query]))

        # find valid candidates for good sample by going through sorted list
        # in reverse and finding index of first sample with at least margin coviews
        # note: this is another way to write candidates_pos = [i for i in candidates if i[1] > margin]
        # but is much faster for large candidate arrays
        for j in range(len(candidates) - 1, -1, -1):
            if candidates[j][1] > margin + candidates[-1][1]:
                candidates_pos = candidates[0:j + 1]
                break
            else:
                candidates_pos = []

        # if there are no valid candidates for good rec, return None to trigger query resample
        if candidates and len(candidates_pos) > 0:
            easy_samples: List = []
            for i in range(n_easy_samples):
                pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random good sample from candidates
                neg = candidates_zero[np.random.randint(len(candidates_zero))]  # random zero
                easy_samples.append([query, pos, (neg, float("-inf"))])
            results.extend(easy_samples)

    return results


def generate_triplets(paper_ids, coviews, margin_fraction, samples_per_query, ratio_hard_negatives, query_ids, data_subset=None, n_jobs=1):
    global _coviews
    global _margin_fraction
    global _samples_per_query
    global _ratio_hard_negatives
    global _query_ids
    global _paper_ids_set

    _coviews = coviews
    _margin_fraction = margin_fraction
    _samples_per_query = samples_per_query
    _ratio_hard_negatives = ratio_hard_negatives
    _query_ids = query_ids
    _paper_ids_set = set(paper_ids)

    logger.info(f'generating triplets with: samples_per_query:{_samples_per_query},'
                f'ratio_hard_negatives:{_ratio_hard_negatives}, margin_fraction:{_margin_fraction}')
    if n_jobs == 1:
        results = [_get_triplet(query) for query in tqdm.tqdm(query_ids)]
    elif n_jobs > 0:
        logger.info(f'running {n_jobs} parallel jobs to get triplets for {data_subset or "not-specified"} set')
        with Pool(n_jobs) as p:
            # results = p.imap(_get_triplet, query, coviews, margin_fraction, paper_ids_set, samples_per_query, ratio_hard_negatives)
            results = list(tqdm.tqdm(p.imap(_get_triplet, query_ids), total=len(query_ids)))
    else:
        raise RuntimeError(f"bad argument `n_jobs`={n_jobs}, `n_jobs` should be -1 or >0")
    for res in results:
        if res:
            for triplet in res:
                yield triplet
