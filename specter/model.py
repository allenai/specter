from typing import Dict, Optional, Union

import numpy
from allennlp.common.checks import ConfigurationError
from overrides import overrides
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch.nn.modules.distance import CosineSimilarity

from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, Seq2SeqEncoder, TextFieldEmbedder, FeedForward, TimeDistributed, LayerNorm
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.modules.token_embedders import Embedding

import logging
import contextlib

logging.basicConfig()
logger = logging.getLogger()


import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    """

    def __init__(self, margin=1.0, distance='l2-norm', reduction='mean'):
        """
        Args:
            margin: margin (float, optional): Default: `1`.
            distance: can be `l2-norm` or `cosine`, or `dot`
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction

    def forward(self, query, positive, negative):
        if self.distance == 'l2-norm':
            distance_positive = F.pairwise_distance(query, positive)
            distance_negative = F.pairwise_distance(query, negative)
            losses = F.relu(distance_positive - distance_negative + self.margin)
        elif self.distance == 'cosine':  # independent of length
            distance_positive = F.cosine_similarity(query, positive)
            distance_negative = F.cosine_similarity(query, negative)
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        elif self.distance == 'dot':  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0],)
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0],)
            losses = F.relu(-distance_positive + distance_negative + self.margin)
        else:
            raise TypeError(f"Unrecognized option for `distance`:{self.distance}")

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise TypeError(f"Unrecognized option for `reduction`:{self.reduction}")


class BinaryLoss(nn.Module):
    """
    Computes contrastive loss[1, 2] twice, one time for the distance between query and positive example,
        and another for the distance between query and negative example. Both use l2-distance.
    [1] http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf, equation 4
    [2] https://gist.github.com/harveyslash/725fcc68df112980328951b3426c0e0b#file-contrastive-loss-py
    """

    def __init__(self, margin=1.0):
        """
        Args:
            margin: margin (float, optional): Default: `1.0`.
        """
        super(BinaryLoss, self).__init__()
        self.margin = margin

    def forward(self, query, positive, negative):
        distance_positive = F.pairwise_distance(query, positive)
        distance_negative = F.pairwise_distance(query, negative)

        # adding the two contrastive losses then simplify
        return torch.pow(distance_positive, 2) + torch.pow(torch.clamp(self.margin - distance_negative, min=0.0), 2)


@Model.register("specter")
class Specter(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 abstract_encoder: Seq2VecEncoder,
                 venue_encoder: Seq2VecEncoder,
                 body_encoder: Seq2VecEncoder = None,
                 predict_mode: bool = False,
                 author_text_embedder: TextFieldEmbedder = None,
                 venue_field_embedder: TextFieldEmbedder = None,
                 author_text_encoder: Seq2VecEncoder = None,
                 # author_id_embedder: Optional[Embedding] = None,
                 author_id_embedder: TextFieldEmbedder = None,
                 # author_position_embedder: Optional[Embedding] = None,
                 author_position_embedder: TextFieldEmbedder = None,
                 feedforward: FeedForward = None,
                 author_feedforward: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 max_num_authors: Optional[int] = 5,
                 dropout: Optional[float] = None,
                 ignore_authors: Optional[bool] = False,
                 layer_norm: Optional[bool] = True,
                 embedding_layer_norm: Optional[bool] = False,
                 loss_distance: Optional[str] = 'l2-norm',
                 loss_margin: Optional[float] = 1,
                 bert_finetune: Optional[bool] = False,
                 include_venue: Optional[bool] = False
                 ) -> None:
        super(Specter, self).__init__(vocab, regularizer)

        for lbl in range(max_num_authors): # TODO SHOULD BE REMOVED
            vocab.add_token_to_namespace(token=str(lbl), namespace='author_positions')

        self.text_field_embedder = text_field_embedder
        self.venue_field_embedder = venue_field_embedder
        self.title_encoder = title_encoder
        self.abstract_encoder = abstract_encoder
        self.body_encoder = body_encoder
        self.venue_encoder = venue_encoder

        self.predict_mode = predict_mode

        self.feedforward = feedforward

        if loss_distance == 'l2-norm':
            self.loss = torch.nn.TripletMarginLoss(margin=loss_margin, reduction='none')
        elif loss_distance == 'binary':
            self.loss = BinaryLoss(margin=loss_margin)
        else:
            self.loss = TripletLoss(margin=loss_margin, distance=loss_distance, reduction='none')

        if layer_norm:
            self.layer_norm = LayerNorm(self.feedforward.get_output_dim())
        self.do_layer_norm = layer_norm

        # self.layer_norm_author_embedding = LayerNorm(author_feedforward.get_output_dim())

        if embedding_layer_norm:
            self.layer_norm_word_embedding = LayerNorm(self.title_encoder.get_input_dim())
            self.layer_norm_word_embedding_venue = LayerNorm(self.venue_encoder.get_input_dim())
        self.embedding_layer_norm = embedding_layer_norm

        self.dropout = Dropout()

        self.ignore_authors = ignore_authors

        if not ignore_authors:
            self.author_id_embedder = author_id_embedder
            self.author_position_embedder = author_position_embedder
            self.author_text_embedder = author_text_embedder
            self.author_text_encoder = author_text_encoder
            # author representation would be a concatenation of author-id and author-position
            # [batch, num-authors, auth-dim + position-dim]
            # we apply timedistributed mlp on top to make this a:
            # [batch, num-authors, dim]
            self.author_time_dist_ff = TimeDistributed(author_feedforward)

        # internal variable showing that the title/abstract should be encoded with a transformer
        # do not change this as it should be by default `false` in this class
        # in the inheriting `PaperRepresentationTransoformer` class it is set to true in the constructor
        # to indicate that the title/abstract should be encoded with a transformer.
        self.tansformer_encoder = False

        self.bert_finetune = bert_finetune
        self.include_venue = include_venue

        self.include_venue = include_venue

        initializer(self)

    def get_embedding_and_mask(self, text_field, embedder_type='generic'):
        if embedder_type == 'author':
            embedded_ = self.author_text_embedder(text_field)
        elif embedder_type == 'venue':
            embedded_ = self.venue_field_embedder(text_field)
        elif embedder_type == 'generic':
            embedded_ = self.text_field_embedder(text_field)
        else:
            raise TypeError(f"Unknown embedder type passed: {embedder_type}")
        mask_ = util.get_text_field_mask(text_field)
        return embedded_, mask_

    def _embed_paper(self,
                    title: torch.Tensor,
                    abstract: torch.Tensor,
                    year: torch.Tensor,
                    venue: torch.Tensor,
                    body: torch.Tensor,
                    author_text: torch.Tensor):
        """ Embed the paper"""

        # in finetuning mode, title and abstract are one long sequence.
        embedded_title, title_mask = self.get_embedding_and_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        if self.dropout:
            encoded_title = self.dropout(encoded_title)

        if self.include_venue:
            embedded_venue, venue_mask = self.get_embedding_and_mask(venue, embedder_type='venue')
            if self.embedding_layer_norm:
                embedded_venue = self.layer_norm_word_embedding_venue(embedded_venue)
            encoded_venue = self.venue_encoder(embedded_venue, venue_mask)
            if self.dropout:
                encoded_venue = self.dropout(encoded_venue)

            paper_embedding = torch.cat((encoded_title, encoded_venue), -1)
        else:
            paper_embedding = encoded_title

        return paper_embedding


    @overrides
    def forward(self,  # type: ignore
                source_title: Dict[str, torch.LongTensor],
                source_abstract: Dict[str, torch.LongTensor] = None,
                source_authors: Dict[str, torch.LongTensor] = None,
                source_author_positions: Dict[str, torch.LongTensor] = None,
                source_year: Dict[str, torch.LongTensor] = None,
                source_venue: Dict[str, torch.LongTensor] = None,
                source_body: Dict[str, torch.LongTensor] = None,
                source_author_text: Dict[str, torch.LongTensor] = None,
                source_paper_id: Dict[str, torch.LongTensor] = None,
                pos_title: Dict[str, torch.LongTensor] = None,
                pos_abstract: Dict[str, torch.LongTensor] = None,
                pos_authors: Dict[str, torch.LongTensor] = None,
                pos_author_positions: Dict[str, torch.LongTensor] = None,
                pos_year: Dict[str, torch.LongTensor] = None,
                pos_venue: Dict[str, torch.LongTensor] = None,
                pos_body: Dict[str, torch.LongTensor] = None,
                pos_author_text: Dict[str, torch.LongTensor] = None,
                pos_paper_id: Dict[str, torch.LongTensor] = None,
                neg_title: Dict[str, torch.LongTensor] = None,
                neg_abstract: Dict[str, torch.LongTensor] = None,
                neg_authors: Dict[str, torch.LongTensor] = None,
                neg_author_positions: Dict[str, torch.LongTensor] = None,
                neg_year: Dict[str, torch.LongTensor] = None,
                neg_venue: Dict[str, torch.LongTensor] = None,
                neg_body: Dict[str, torch.LongTensor] = None,
                neg_author_text: Dict[str, torch.LongTensor] = None,
                neg_paper_id: Dict[str, torch.LongTensor] = None,
                data_source: Optional[str] = None,
                mixing_ratio: Dict[str, torch.LongTensor] = None,
                dataset: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            Most args are self explanatory

            data_source: in case of multitask data reader, this shows which dataset
                the instance is coming from
            mixing_ratio: in case of multitask data reader and when in multitask_data_reader the
                `use_loss_weighting=True`, this specifies the corresponding loss weight
                for this instance. It is the loss weight for the corresponding dataset and it
                is proportional to the square root of the size of the dataset.
                See `multitask_data_reader.py` for more details
            dataset: AllenNLPs interleaving dataset reader adds a `dataset` metadatafield
                So we need to have it here
        """
        src_paper = self._embed_paper(source_title,
                                    source_abstract,
                                    source_year,
                                    source_venue, source_body, source_author_text)

        if (pos_title is None or neg_title is None) and not self.predict_mode:
            raise ConfigurationError('Positive or negative paper title is None in training mode. This field is'
                                     ' mandatory for training. If in prediction mode, '
                                     'set `predict_mode`=true in config')

        # this will be training mode, embed the positive and negative papers
        if pos_title is not None and neg_title is not None and not self.predict_mode:
            # embed the positive paper
            pos_paper = self._embed_paper(pos_title,
                                          pos_abstract,
                                          pos_year,
                                          pos_venue, pos_body, pos_author_text)

            # embed the negative paper
            neg_paper = self._embed_paper(neg_title,
                                          neg_abstract,
                                          neg_year,
                                          neg_venue, neg_body, neg_author_text)

            loss = self.loss(src_paper, pos_paper, neg_paper)

            if mixing_ratio is not None:
                loss *= mixing_ratio.float()
                loss = loss.mean()
            else:
                loss = loss.mean()

            output_dict = {"loss": loss}

        else:  # predict mode, we only care about the source paper
            output_dict = {"embedding": src_paper}

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict
