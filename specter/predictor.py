"""
Given a paper and a pre-trained model, it embeds and returns it
"""
import json
from typing import Dict, List

from allennlp.common import Params
from allennlp.common.util import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model, Archive
from allennlp.predictors.predictor import Predictor
from overrides import overrides
import numpy as np


@Predictor.register('specter_predictor')
class SpecterPredictor(Predictor):

    @overrides
    def predict_json(self, paper: JsonDict) -> Dict:
        ret = {}
        for key in ['paper_id', 'title', 'abstract', 'authors', 'venue']:
            try:
                ret[key] = paper[key]
            except KeyError:
                pass
        ret['embedding'] = []
        try:
            if hasattr(self._model, 'bert_finetune') and self._model.bert_finetune:
                if not paper['title'] and not paper['abstract']:
                    return ret
            else:
                if not paper['title'] or not paper['abstract']:
                    return ret
        except KeyError:
            return ret

        self._dataset_reader.text_to_instance(paper)

        instance = self._dataset_reader.text_to_instance(paper)

        outputs = self._model.forward_on_instance(instance)

        ret['embedding'] = outputs['embedding'].tolist()
        return ret


    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = []
        skipped_idx = []
        for idx, json_dict in enumerate(inputs):
            paper = {}
            if 'title' not in json_dict:
                skipped_idx.append(idx)
                continue
            skip = False
            for key in ['paper_id', 'title', 'abstract', 'authors', 'venue']:
                try:
                    paper[key] = json_dict[key]
                except KeyError:
                    pass
            paper['embedding'] = []
            try:
                if hasattr(self._model, 'bert_finetune') and self._model.bert_finetune:
                    # this model concatenates title/abstract
                    if not json_dict['title']:
                        skip = True
                else:
                    # both title and abstract must be present
                    if not json_dict['title'] or not json_dict['abstract']:
                        skip = True
            except KeyError:
                skip = True
            if not skip:
                instances.append(self._dataset_reader.text_to_instance(json_dict))
            else:
                skipped_idx.append(idx)
        if instances:
            outputs = self._model.forward_on_instances(instances)
        else:
            outputs = []
        k = 0
        results = []
        for j in range(len(inputs)):
            paper = {}
            for key in ['paper_id', 'title']:
                try:
                    paper[key] = inputs[j][key]
                except KeyError:
                    pass
            paper['embedding'] = []
            if not skipped_idx or k >= len(skipped_idx) or skipped_idx[k] != j:
                paper['embedding'] = outputs[j - k]['embedding'].tolist()
                results.append(paper)
            else:
                paper['embedding'] = []
                results.append(paper)
                k += 1
        return results

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs, cls=NumpyEncoder) + "\n"

    @overrides
    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        if line.strip() not in self._dataset_reader.papers:
            return {'paper_id': line.strip()}
        else:
            return self._dataset_reader.papers[line.strip()]


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
