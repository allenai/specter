"""
The script reads the coview/co-citation data and returns them as triplet format
i.e., ['Query_paper', ('Positive_paper_id', num-coviews), ('Negative_paper_id', num_coviews)
"""
import operator
from typing import List, Tuple, Dict, Optional, Generator, NoReturn, Iterator

import math
import numpy as np
import logging
import tqdm
import random

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


class TripletGenerator:
    """ Class to generate triplets"""

    def __init__(self,
                 paper_ids: List[str],
                 coviews: Dict[str, List[Tuple[str, int]]],
                 margin_fraction: float,
                 samples_per_query: int,
                 ratio_hard_negatives: Optional[float] = 0.5) -> NoReturn:
        """
        Args:
            paper_ids: list of all paper ids
            coviews: a dictionary where keys are paper ids and values are lists of [paper_id, count] pairs
                showing the number of coviews for each paper
            margin_fraction: minimum margin of co-views between positive and negative samples
            samples_per_query: how many samples for each query
            query_list: list of query ids. If None, it will return for all papers in the coviews file
            ratio_hard_negatives: ratio of negative samples selected from difficult and easy negatives
                respectively. Difficult negative samples are those that are also coviewed but less than
                a positive sample. Easy negative samples are those with zero coviews
        """
        self.paper_ids = paper_ids
        self.paper_ids_set = set(paper_ids)
        self.coviews = coviews
        self.margin_fraction = margin_fraction
        self.samples_per_query = samples_per_query
        self.ratio_hard_negatives = ratio_hard_negatives

    def _get_triplet(self, query):
        if query not in self.coviews:
            return
        # self.coviews[query] is a dictionary of format {paper_id: {count: 1, frac: 1}}
        candidates = [(k, v['count']) for k, v in self.coviews[query].items()]
        candidates = sorted(candidates, key=operator.itemgetter(1), reverse=True)
        if len(candidates) > 1:
            coview_spread = candidates[0][1] - candidates[-1][1]
        else:
            coview_spread = 0
        margin = self.margin_fraction * coview_spread  # minimum margin of coviews between good and bad sample

        # If distance is 1 increase margin to 1 otherwise any margin_fraction will pass
        if is_int(candidates[0][1]) and is_int(candidates[-1][1]) and coview_spread == 1:
            margin = np.ceil(margin)

        difficult_triplets = self._get_hard_negatives(query, candidates, margin)
        n_easy_samples = self.samples_per_query - len(difficult_triplets)
        easy_triplets = self._get_easy_negatives(query, candidates, margin, n_easy_samples)
        return difficult_triplets + easy_triplets

    def generate_triplets(self, query_ids: List[str]) -> Iterator[List[Tuple]]:
        """ Generate triplets from a list of query ids

        This generates a list of triplets each query according to:
            [(query_id, (positive_id, coviews), (negative_id, coviews)), ...]
        The upperbound of the list length is according to self.samples_per_query

        Args:
            query_ids: a list of query paper ids

        Returns:
            Lists of tuples
                The format of tuples is according to the triples
        """
        # logger.info('Generating triplets for queries')
        count_skipped = 0  # count how many of the queries are not in coveiws file
        count_success = 0

        for query in query_ids:  # tqdm.tqdm(query_ids):
            results = self._get_triplet(query)
            if results:
                for triplet in results:
                    yield triplet
                count_success += 1
            else:
                count_skipped += 1
        logger.info(f'Done generating triplets, #successful queries: {count_success},'
                    f'#skipped queries: {count_skipped}')

    def _get_easy_negatives(self, query_id: str, candidates: List[Tuple[str, float]], margin: float,
                            n_samples: int) -> \
            List[Tuple[str, Tuple[str, int], Tuple[str, int]]]:
        """ Given a query get easy negative samples
        Easy samples are defined by those that are at with 0 coviews/copdfs/etc.

        Args:
            query_id: string specifying the id of the query paper
            candidates: a list of candidates, i.e., papers with co-view information with the query paper
            margin: minimum distance of coviews between positive and negative example
        """
        # If there are fewer than 2 candidates, return none
        # if len(candidates) < 2 or margin == 0:
        if len(candidates) < 2:
            return []

        # valid candidates are those with zeros
        candidates_zero = list(self.paper_ids_set.difference([i[0] for i in candidates] + [query_id]))

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
        if len(candidates_pos) == 0:
            return []

        easy_samples: List = []
        for i in range(n_samples):
            pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random good sample from candidates
            neg = candidates_zero[np.random.randint(len(candidates_zero))]  # random zero
            easy_samples.append([query_id, pos, (neg, float("-inf"))])

        return easy_samples

    def _get_hard_negatives(self, query_id: str, candidates: List[Tuple[str, float]], margin: float) ->\
            List[Tuple[str, Tuple[str, int], Tuple[str, int]]]:
        """ Given a query get difficult negative samples
        hard/difficult samples are defined by those samples that have fewer coviews than the positive ones

        Args:
            query_id: string specifying the id of the query paper
            candidates: a list of candidates, i.e., papers with co-view information with the query paper
            margin: minimum distance of coviews between positive and negative example
        """
        # If there are fewer than 2 candidates, return none
        if len(candidates) < 2 or margin == 0:
            return []

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

        if neg_len == 0:
            return []

        # generate hard candidates
        samples_per_query = math.ceil(self.ratio_hard_negatives * self.samples_per_query)
        # if there aren't enough candidates to generate enough unique samples
        # reduce the number of samples to make it possible for them to be unique
        if (pos_len * neg_len) < samples_per_query:
            samples_per_query = pos_len * neg_len

        hard_samples: List = []
        for i in range(samples_per_query):
            # find the negative sample first.
            neg = candidates_hard_neg[np.random.randint(len(candidates_hard_neg))]  # random neg sample from candidates

            # find the good sample. find valid candidates by going through sorted list
            # in reverse and finding index of first sample with bad sample + margin
            for j in range(len(candidates) - 1, -1, -1):
                if candidates[j][1] > (neg[1] + margin):
                    candidates_pos = candidates[0:j + 1]
                    break
            pos = candidates_pos[np.random.randint(len(candidates_pos))]  # random pos sample from candidates

            # append the good and bad samples with their coview number to output
            hard_samples.append([query_id, pos, neg])

        return hard_samples
