""" Data reader for Co-views data. """
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Iterable
import hashlib
import os
import pathlib
import json
import logging

import numpy as np

import dill

from allennlp.data import TokenIndexer, Tokenizer
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from numpy.compat import os_PathLike
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, _LazyInstances
from allennlp.data.fields import LabelField, TextField, MultiLabelField, ListField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedBertIndexer
from allennlp.data.tokenizers.word_splitter import WordSplitter, SimpleWordSplitter, BertBasicWordSplitter
from allennlp.data.tokenizers.token import Token
from specter.data_utils.create_training_files import get_text_tokens

from specter.data_utils.triplet_sampling import TripletGenerator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# to avoid empty embedding lookup, we need a placeholder to replace no-venue cases
NO_VENUE_TEXT = '--no_venue--'


@DatasetReader.register("specter_data_reader_pickled")
class DataReaderFromPickled(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 word_splitter: WordSplitter = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_sequence_length: int = 256,
                 concat_title_abstract: bool = None
                 ) -> None:
        """
        Dataset reader that uses pickled preprocessed instances
        Consumes the output resulting from data_utils/create_training_files.py

        the additional arguments are not used here and are for compatibility with
        the other data reader at prediction time
        """
        self.max_sequence_length = max_sequence_length
        self.token_indexers = token_indexers
        self._concat_title_abstract = concat_title_abstract
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str):
        """
        Args:
            file_path: path to the pickled instances
        """
        with open(file_path, 'rb') as f_in:
            unpickler = pickle.Unpickler(f_in)
            while True:
                try:
                    instance = unpickler.load()
                    # compatibility with old models:
                    # for field in instance.fields:
                    #     if hasattr(instance.fields[field], '_token_indexers') and 'bert' in instance.fields[field]._token_indexers:
                    #         if not hasattr(instance.fields['source_title']._token_indexers['bert'], '_truncate_long_sequences'):
                    #             instance.fields[field]._token_indexers['bert']._truncate_long_sequences = True
                    #             instance.fields[field]._token_indexers['bert']._token_min_padding_length = 0
                    if self.max_sequence_length:
                        for paper_type in ['source', 'pos', 'neg']:
                            if self._concat_title_abstract:
                                tokens = []
                                title_field = instance.fields.get(f'{paper_type}_title')
                                abst_field = instance.fields.get(f'{paper_type}_abstract')
                                if title_field:
                                    tokens.extend(title_field.tokens)
                                if tokens:
                                    tokens.extend([Token('[SEP]')])
                                if abst_field:
                                    tokens.extend(abst_field.tokens)
                                if title_field:
                                    title_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = title_field
                                elif abst_field:
                                    abst_field.tokens = tokens
                                    instance.fields[f'{paper_type}_title'] = abst_field
                                else:
                                    yield None
                                # title_tokens = get_text_tokens(query_title_tokens, query_abstract_tokens, abstract_delimiter)
                                # pos_title_tokens = get_text_tokens(pos_title_tokens, pos_abstract_tokens, abstract_delimiter)
                                # neg_title_tokens = get_text_tokens(neg_title_tokens, neg_abstract_tokens, abstract_delimiter)
                                # query_abstract_tokens = pos_abstract_tokens = neg_abstract_tokens = []
                            for field_type in ['title', 'abstract', 'authors', 'author_positions']:
                                field = paper_type + '_' + field_type
                                if instance.fields.get(field):
                                    instance.fields[field].tokens = instance.fields[field].tokens[:self.max_sequence_length]
                                if field_type == 'abstract' and self._concat_title_abstract:
                                    instance.fields.pop(field, None)
                    yield instance
                except EOFError:
                    break




@DatasetReader.register("specter_data_reader")
class DataReader(DatasetReader):

    def __init__(self,
                 lazy: bool = False,
                 paper_features_path: str = None,
                 word_splitter: WordSplitter = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 data_file: Optional[str] = None,
                 samples_per_query: int = 5,
                 margin_fraction: float = 0.5,
                 ratio_hard_negatives: float = 0.5,
                 predict_mode: bool = False,
                 max_num_authors: Optional[int] = 5,
                 ratio_training_samples: Optional[float] = None,
                 max_sequence_length: Optional[int] = -1,
                 cache_path: Optional[str] = None,
                 overwrite_cache: Optional[bool] = False,
                 use_cls_token: Optional[bool] = None,
                 concat_title_abstract: Optional[bool] = None,
                 coviews_file: Optional[str] = None,
                 included_text_fields: Optional[str] = None,
                 use_paper_feature_cache: bool = True
                 ) -> None:
        """
        Args:
            lazy: if false returns a list
            paper_features_path: path to the paper features json file (result of scripts.generate_paper_features.py
            candidates_path: path to the candidate papers
            tokenizer: tokenizer to be used for tokenizing strings
            token_indexers: token indexer for indexing vocab
            data_file: path to the data file (e.g, citations)
            samples_per_query: number of triplets to generate for each query
            margin_fraction: minimum margin of co-views between positive and negative samples
            ratio_hard_negatives: ratio of training data that is selected from hard negatives
                remaining is allocated to easy negatives. should be set to 1.0 in case of similar click data
            predict_mode: if `True` the model only considers the current paper and returns an embedding
                otherwise the model uses the triplet format to train the embedder
            author_id_embedder: Embedder for author ids
            s2_id_embedder: Embedder for respresenting s2 ids
            other_id_embedder: Embedder for representing other ids (e.g., id assigned by metadata)
            max_num_authors: maximum number of authors,
            ratio_training_samples: Limits training to proportion of all training instances
            max_sequence_length: Longer sequences would be truncated (if -1 then there would be no truncation)
            cache_path: Path to file to cache instances, if None, instances won't be cached.
                If specified, instances are cached after being created so next time they are not created
                again from scratch
            overwrite_cache: If true, it overwrites the cached files. Each file corresponds to
                all instances created from the train, dev or test set.
            use_cls_token: Like bert, use an additional CLS token in the begginning (for transoformer)
            concat_title_abstract: Whether to consider title and abstract as a single field.
            coviews_file: Only for backward compatibility to work with older models (renamed to 
                `data_file` in newer models), leave this empty as it won't have any effect
            included_text_fields: space delimited fields to concat to the title: e.g., `title abstract authors`
            use_paper_feature_cache: set to False to disable the in-memory cache of paper features
        """
        super().__init__(lazy)
        self._word_splitter = word_splitter or SimpleWordSplitter()
        self._tokenizer = tokenizer or WordTokenizer(self._word_splitter)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._token_indexer_author_id = {"tokens": SingleIdTokenIndexer(namespace='author')}
        self._token_indexer_author_position = \
            {"tokens": SingleIdTokenIndexer(namespace='author_positions')}

        self._token_indexer_venue = {"tokens": SingleIdTokenIndexer(namespace='venue')}
        self._token_indexer_id = {"tokens": SingleIdTokenIndexer(namespace='id')}

        with open(paper_features_path) as f_in:
            self.papers = json.load(f_in)
        self.samples_per_query = samples_per_query
        self.margin_fraction = margin_fraction
        self.ratio_hard_negatives = ratio_hard_negatives

        self.predict_mode = predict_mode
        self.max_sequence_length = max_sequence_length
        self.use_cls_token = use_cls_token

        if data_file and not predict_mode:
            # logger.info(f'reading contents of the file at: {coviews_file}')
            with open(data_file) as f_in:
                self.dataset = json.load(f_in)
            # logger.info(f'reading complete. Total {len(self.dataset)} records found.')
            root_path, _ = os.path.splitext(data_file)
            # for multitask interleaving reader, track which dataset the instance is coming from
            self.data_source = root_path.split('/')[-1]
        else:
            self.dataset = None
            self.data_source = None

        self.max_num_authors = max_num_authors

        self.triplet_generator = TripletGenerator(
            paper_ids=list(self.papers.keys()),
            coviews=self.dataset,
            margin_fraction=margin_fraction,
            samples_per_query=samples_per_query,
            ratio_hard_negatives=ratio_hard_negatives
        )
        self.paper_feature_cache = {}  # paper_id -> paper features. Serves as a cache for the _get_paper_features function

        self.ratio_training_samples = float(ratio_training_samples) if ratio_training_samples else None

        self.cache_path = cache_path
        self.overwrite_cache = overwrite_cache
        self.data_file = data_file
        self.paper_features_path = paper_features_path
        self.ratio_training_samples = ratio_training_samples

        self.concat_title_abstract = concat_title_abstract
        self.included_text_fields = set(included_text_fields.split())
        self.use_paper_feature_cache = use_paper_feature_cache

        self.abstract_delimiter = [Token('[SEP]')]
        self.author_delimiter = [Token('[unused0]')]


    def _get_paper_features(self, paper: Optional[dict] = None) -> Tuple[List[Token], List[Token], List[Token], int, List[Token]]:
        """ Given a paper, extract and tokenize abstract, title, venue and year"""
        if paper:
            paper_id = paper.get('paper_id')

            # This function is being called by the same paper multiple times.
            # Cache the result to avoid wasted compute
            if self.use_paper_feature_cache and paper_id in self.paper_feature_cache: 
                return self.paper_feature_cache[paper_id]

            if not self.concat_title_abstract:
                abstract_tokens = self._tokenizer.tokenize(paper.get('abstract') or '')
                title_tokens = self._tokenizer.tokenize(paper.get('title') or '')
                if self.max_sequence_length > 0:
                    title_tokens = title_tokens[:self.max_sequence_length]
                    abstract_tokens = abstract_tokens[:self.max_sequence_length]
            else:
                abstract_tokens = self._tokenizer.tokenize(paper.get("abstract") or "")
                title_tokens = self._tokenizer.tokenize(paper.get("title") or "")
                if 'abstract' in self.included_text_fields:
                    title_tokens = get_text_tokens(title_tokens, abstract_tokens, self.abstract_delimiter)
                if 'authors' in self.included_text_fields:
                    author_text = ' '.join(paper.get("author-names") or [])
                    author_tokens = self._tokenizer.tokenize(author_text)
                    max_seq_len_title = self.max_sequence_length - 15  # reserve max 15 tokens for author names
                    title_tokens = title_tokens[:max_seq_len_title] + self.author_delimiter + author_tokens
                title_tokens = title_tokens[:self.max_sequence_length]
                # abstract and title are identical (abstract won't be used in this case)
                abstract_tokens = title_tokens

            venue = self._tokenizer.tokenize(paper.get('venue') or NO_VENUE_TEXT)
            year = paper.get('year') or 0
            body_tokens = self._tokenizer.tokenize(paper.get('body')) if 'body' in paper else None
            features = abstract_tokens, title_tokens, venue, year, body_tokens

            if self.use_paper_feature_cache:
                self.paper_feature_cache[paper_id] = features

            return features
        else:
            return None, None, None, None, None

    def _get_author_field(self, authors: List[str]) -> Tuple[ListField, ListField]:
        """
        Get a Label field associated with authors along with their position
        Args:
            authors: list of authors

        Returns:
            authors and their positions
        """
        if authors == []:
            authors = ['##']
        authors = [self._tokenizer.tokenize(author) for author in authors]
        if len(authors) > self.max_num_authors:
            authors = authors[:self.max_num_authors - 1] + [authors[-1]]
        author_field = ListField([TextField(author, token_indexers=self._token_indexer_author_id) for author in authors])

        author_positions = []
        for i, _ in enumerate(authors):
            if i == 0:
                author_positions.append(TextField(
                    self._tokenizer.tokenize('00'), token_indexers=self._token_indexer_author_position))
            elif i < len(authors) - 1:
                author_positions.append(TextField(
                    self._tokenizer.tokenize('01'), token_indexers=self._token_indexer_author_position))
            else:
                author_positions.append(TextField(
                    self._tokenizer.tokenize('02'), token_indexers=self._token_indexer_author_position))
        position_field = ListField(author_positions)
        return author_field, position_field

    def get_hash(self, file_path):
        """
        Get hashname for the current dataset reader config
        """
        key = f"{file_path},{self.data_file},{self.paper_features_path}," \
            f"{self.ratio_training_samples},{str(self._word_splitter.__class__)},{self.samples_per_query}," \
            f"{self.margin_fraction},{self.ratio_hard_negatives},{self.use_cls_token}," \
            f"{self.max_sequence_length}"
        if self.concat_title_abstract:
            key += 'concat-title'
        return hashlib.md5(key.encode('utf-8')).hexdigest()


    @overrides
    def read(self, file_path: str) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        ensures that the result is a list, then returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        In this case your implementation of ``_read()`` must also be lazy
        (that is, not load all instances into memory at once), otherwise
        you will get a ``ConfigurationError``.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        lazy = getattr(self, 'lazy', None)
        if lazy is None:
            logger.warning("DatasetReader.lazy is not set, "
                           "did you forget to call the superclass constructor?")
        if lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            if self.cache_path is not None:
                # create a key for the file based on the reader config
                hash_ = self.get_hash(file_path)
                pathlib.Path(self.cache_path).mkdir(parents=True, exist_ok=True)
                cache_file = os.path.join(self.cache_path, (hash_ + '.cache'))
                if not os.path.exists(cache_file) or self.overwrite_cache:
                    instances = self._read(file_path)
                    if not isinstance(instances, list):
                        instances = [instance for instance in Tqdm.tqdm(instances)]
                    if not instances:
                        raise ConfigurationError("No instances were read from the given filepath {}. "
                                                 "Is the path correct?".format(file_path))
                    logger.info(f'caching instances to file: {cache_file}')

                    with open(cache_file, 'wb') as cache:
                        dill.dump(instances, cache)
                else:
                    logger.info(f'Reading instances from cache file: {cache_file}')
                    # instances = []
                    # with open(cache_file, 'rb') as cache:
                    #     start   = time.time()
                    #     instances = []
                    #     for line in Tqdm.tqdm(cache):
                    #         instances.append(self.deserialize_instance(line.strip()))
                    #     print(time.time()-start)
                    with open(cache_file, 'rb') as f_in:
                        instances = dill.load(f_in)
            else:
                instances = self._read(file_path)
                if not isinstance(instances, list):
                    instances = [instance for instance in Tqdm.tqdm(instances)]
                if not instances:
                    raise ConfigurationError("No instances were read from the given filepath {}. "
                                             "Is the path correct?".format(file_path))
            return instances

    @overrides
    def _read(self, query_file: str):
        """
        Args:
            query_file: path to the list of query paper ids, can be train/dev/test
        """
        query_ids = [line.strip() for line in open(query_file)]

        # logger.info('reading triplets ...')
        if self.ratio_training_samples is not None:
            total_len = len(query_ids)
            query_ids = query_ids[:int(len(query_ids) * self.ratio_training_samples)]
            logger.info(f'using {len(query_ids)} (={len(query_ids)*100/total_len:.4f}%) '
                        f'out of total {total_len} available query ids')

        count_success, count_fail = 0, 0

        logger.info('reading triplets ...')

        # triplets are in format (p0, (p1, count1), (p2, count2))
        for triplet in self.triplet_generator.generate_triplets(query_ids):
            try:
                source_paper = self.papers[triplet[0]]
                pos_paper = self.papers[triplet[1][0]]
                neg_paper = self.papers[triplet[2][0]]
                count_success += 1

                # check if all papers have title and abstract
                failed = False
                for paper in (source_paper, pos_paper, neg_paper):
                    if not paper['abstract'] or not paper['title']:
                        failed = True
                        break
                if failed:
                    count_fail += 1
                    continue

                yield (self.text_to_instance(
                    source_paper, pos_paper, neg_paper, self.data_source))
            except KeyError:
                count_fail += 1
                pass
        logger.info(f'done reading triplets success: {count_success}, failed: {count_fail}')

    @overrides
    def text_to_instance(self,
                         source_paper: dict,
                         positive_paper: Optional[dict] = None,
                         negative_paper: Optional[dict] = None,
                         data_source: Optional[str] = None,
                         mixing_ratio: Optional[float] = None) -> Instance:

        source_abstract_tokens, source_title_tokens, source_venue, source_year, source_body =\
            self._get_paper_features(source_paper)
        pos_abstract_tokens, pos_title_tokens, pos_venue, pos_year, pos_body = \
            self._get_paper_features(positive_paper)
        neg_abstract_tokens, neg_title_tokens, neg_venue, neg_year, neg_body = \
            self._get_paper_features(negative_paper)

        source_author_tokens = None
        pos_author_tokens = None
        neg_author_tokens = None

        source_author, source_author_position = self._get_author_field(source_paper['authors'])
        if positive_paper is not None:
            pos_author, pos_author_position = self._get_author_field(positive_paper['authors'])
        if negative_paper is not None:
            neg_author, neg_author_position = self._get_author_field(negative_paper['authors'])
        fields = {
            'source_abstract': TextField(source_abstract_tokens, self._token_indexers),
            'source_title': TextField(source_title_tokens, self._token_indexers),
            'source_authors': source_author,
            'source_author_positions': source_author_position,
            'source_year': LabelField(source_year, skip_indexing=True, label_namespace='year'),
            'source_venue': TextField(source_venue, self._token_indexer_venue),
            'source_paper_id': MetadataField(source_paper['paper_id']),
        }

        if source_author_tokens is not None:
            fields['source_author_text'] = TextField(source_author_tokens, self._token_indexers)
        if positive_paper:
            fields['pos_abstract'] = TextField(pos_abstract_tokens, self._token_indexers)
            fields['pos_title'] = TextField(pos_title_tokens, self._token_indexers)
            fields['pos_authors'] = pos_author
            fields['pos_author_positions'] = pos_author_position
            if pos_author_tokens is not None:
                fields['pos_author_text'] = pos_author_tokens
            fields['pos_year'] = LabelField(pos_year, skip_indexing=True, label_namespace='year')
            fields['pos_venue'] = TextField(pos_venue, self._token_indexer_venue)
            fields['pos_paper_id'] = MetadataField(positive_paper['paper_id'])
        if negative_paper:
            fields['neg_abstract'] = TextField(neg_abstract_tokens, self._token_indexers)
            fields['neg_title'] = TextField(neg_title_tokens, self._token_indexers)
            fields['neg_authors'] = neg_author
            fields['neg_author_positions'] = neg_author_position
            if neg_author_tokens is not None:
                fields['neg_author_text'] = neg_author_tokens
            fields['neg_year'] = LabelField(neg_year, skip_indexing=True, label_namespace='year')
            fields['neg_venue'] = TextField(neg_venue, self._token_indexer_venue)
            fields['neg_paper_id'] = MetadataField(negative_paper['paper_id'])
        if data_source:
            fields['data_source'] = MetadataField(data_source)
        if mixing_ratio is not None:
            fields['mixing_ratio'] = ArrayField(mixing_ratio)
        return Instance(fields)
