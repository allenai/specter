"""
Slightly modifies AllenNLPs `commands.predict.py` to allow using base_reader
instead of `multiprocess`.
This is for cases where the archived model is trained with multiprocess reader
Main modification is the `predictor_from_archive` function
"""
from itertools import islice
from typing import List, Iterator, Optional, Dict, Any
import argparse
import sys
import json

from allennlp import __version__
from allennlp.commands import ArgumentParserWithDefaults
from allennlp.commands.predict import _PredictManager
from allennlp.commands.subcommand import Subcommand
from allennlp.common import Params, Tqdm
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.params import parse_overrides
from allennlp.common.util import lazy_groups_of, import_submodules
from allennlp.models import Archive
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict, DEFAULT_PREDICTORS
from allennlp.data import Instance, DatasetReader
from overrides import overrides
from tqdm import tqdm


class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=str, help='path to input file')

        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--use-dataset-reader',
                               action='store_true',
                               help='Whether to use the dataset reader of the original model to load Instances')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser


class _PredictManagerCustom(_PredictManager):
    """
    Extends the following functions from allennlp's _PredictManager class
    `run` function to print predict progress
    """

    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool) -> None:
        super(_PredictManagerCustom, self).__init__(predictor, input_file, output_file, batch_size, print_to_console,
                                                    has_dataset_reader)
        self.total_size = int(sum([1 for _ in open(self._input_file)]) / self._batch_size)

    @overrides
    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        if has_reader:
            for batch in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size), total=self.total_size, unit="batches"):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    index = index + 1
        else:
            for batch_json in tqdm(lazy_groups_of(self._get_json_data(), self._batch_size), total=self.total_size, unit="batches"):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(index, result, json.dumps(model_input_json))
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()



def predictor_from_archive(archive: Archive, predictor_name: str = None,
                           paper_features_path: str = None) -> 'Predictor':
    """
    Extends allennlp.predictors.predictor.from_archive to allow processing multiprocess reader

    paper_features_path is passed to replace the correct one if the dataset_reader is multiprocess
    """

    # Duplicate the config so that the config inside the archive doesn't get consumed
    config = archive.config.duplicate()

    if not predictor_name:
        print("[DEBUG] no predictor name")
        model_type = config.get("model").get("type")
        if not model_type in DEFAULT_PREDICTORS:
            raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
                                     f"Please specify a predictor explicitly.")
        print("[DEBUG] default predictors: ", DEFAULT_PREDICTORS)
        predictor_name = DEFAULT_PREDICTORS[model_type]

    dataset_config = config["dataset_reader"].as_dict()
    if dataset_config['type'] == 'multiprocess':
        dataset_config = dataset_config['base_reader']
        if paper_features_path:
            dataset_config['paper_features_path'] = paper_features_path
        dataset_reader_params = Params(dataset_config)

    else:
        dataset_reader_params = config["dataset_reader"]

    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    model = archive.model
    model.eval()

    return Predictor.by_name(predictor_name)(model, dataset_reader)


def _get_predictor(args: argparse.Namespace) -> Predictor:
    check_for_gpu(args.cuda_device)
    print("[DEBUG]","args.weights_file:", args.weights_file)
    print("[DEBUG]", "args.overrides", args.overrides)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)
    ov = parse_overrides(args.overrides)
    paper_features_path = None
    try:
        paper_features_path = ov['dataset_reader']['paper_features_path']
    except KeyError:
        pass
    return predictor_from_archive(archive, args.predictor, paper_features_path)


def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManagerCustom(predictor,
                                    args.input_file,
                                    args.output_file,
                                    args.batch_size,
                                    not args.silent,
                                    args.use_dataset_reader)
    manager.run()


def main(prog: str = None,
         subcommand_overrides: Dict[str, Subcommand] = {}) -> None:
    """
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = ArgumentParserWithDefaults(description="Run AllenNLP", usage='%(prog)s', prog=prog)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)

    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            # Default commands
            "predict": Predict(),
            # Superseded by overrides
            **subcommand_overrides
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != "configure":
            subparser.add_argument('--include-package',
                                   type=str,
                                   action='append',
                                   default=[],
                                   help='additional packages to include')

    args = parser.parse_args()
    # print("DEBUG ---args:", args)
    # sys.exit()
    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, 'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()


def run():
    main(prog="allennlp")


if __name__ == "__main__":
    run()
