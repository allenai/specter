# basic python packages
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
import torch.nn.functional as F
import torch.nn as nn
import torch
import json
import pickle
from typing import Dict
import argparse
from argparse import Namespace
import glob
import random
import numpy as np
import itertools
import requests
import logging
import os
import traceback
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# pytorch packages

# pytorch lightning packages

# huggingface transformers packages


# Globe constants
training_size = 684100
# validation_size = 145375

# log_every_n_steps how frequently pytorch lightning logs.
# By default, Lightning logs every 50 rows, or 50 training steps.
log_every_n_steps = 1

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


"""
自分で定義するデータセット
"""


class MyData(IterableDataset):
    # def __init__(self, data, tokenizer, size):
    def __init__(self, data, labeled_abst_Dict, label, tokenizer, block_size=100):
        self.data_instances = data
        self.labeled_abst_dict = labeled_abst_Dict
        self.label = label
        self.tokenizer = tokenizer
        self.block_size = block_size

        # self.size = size
        # self.max_length = max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # for data_instance in self.data_instances:
        #         yield self.retTransformersInput(data_instance, self.tokenizer)

        if worker_info is None:
            for data_instance in self.data_instances:
                yield self.retTransformersInput(data_instance, self.tokenizer)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            i = 0
            for data_instance in self.data_instances:
                if int(i / self.block_size) % num_workers != worker_id:
                    i = i + 1
                    pass
                else:
                    i = i + 1
                    yield self.retTransformersInput(data_instance, self.tokenizer)

    def retTransformersInput(self, data_instance, tokenizer):
        sourceEncoded = self.tokenizer(
            self.labeled_abst_dict[data_instance["source"]][self.label],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        posEncoded = self.tokenizer(
            self.labeled_abst_dict[data_instance["pos"]][self.label],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        negEncoded = self.tokenizer(
            self.labeled_abst_dict[data_instance["neg"]][self.label],
            padding="max_length",
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        source_input = {
            'input_ids': sourceEncoded["input_ids"][0],
            'attention_mask': sourceEncoded["attention_mask"][0],
            'token_type_ids': sourceEncoded["token_type_ids"][0]
        }
        pos_input = {
            'input_ids': posEncoded["input_ids"][0],
            'attention_mask': posEncoded["attention_mask"][0],
            'token_type_ids': posEncoded["token_type_ids"][0]
        }
        neg_input = {
            'input_ids': negEncoded["input_ids"][0],
            'attention_mask': negEncoded["attention_mask"][0],
            'token_type_ids': negEncoded["token_type_ids"][0]
        }
        return source_input, pos_input, neg_input


"""
ロス計算を行うモジュール
"""


class TripletLoss(nn.Module):
    """
    Triplet loss: copied from  https://github.com/allenai/specter/blob/673346f9f76bcf422b38e0d1b448ef4414bcd4df/specter/model.py#L159 without any change
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
            losses = F.relu(distance_positive -
                            distance_negative + self.margin)
        elif self.distance == 'cosine':  # independent of length
            distance_positive = F.cosine_similarity(query, positive)
            distance_negative = F.cosine_similarity(query, negative)
            losses = F.relu(-distance_positive +
                            distance_negative + self.margin)
        elif self.distance == 'dot':  # takes into account the length of vectors
            shapes = query.shape
            # batch dot product
            distance_positive = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                positive.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            distance_negative = torch.bmm(
                query.view(shapes[0], 1, shapes[1]),
                negative.view(shapes[0], shapes[1], 1)
            ).reshape(shapes[0], )
            losses = F.relu(-distance_positive +
                            distance_negative + self.margin)
        else:
            raise TypeError(
                f"Unrecognized option for `distance`:{self.distance}")

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise TypeError(
                f"Unrecognized option for `reduction`:{self.reduction}")


"""
モデルのクラス
"""


class Specter(pl.LightningModule):
    def __init__(self, init_args):
        super().__init__()
        if isinstance(init_args, dict):
            # for loading the checkpoint, pl passes a dict (hparams are saved as dict)
            init_args = Namespace(**init_args)
        checkpoint_path = init_args.checkpoint_path
        logger.info(f'loading model from checkpoint: {checkpoint_path}')

        self.label = init_args.label
        self.hparams = init_args

        # self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
        # self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

        # SPECTERを初期値とする場合
        self.model = AutoModel.from_pretrained("allenai/specter")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

        self.tokenizer.model_max_length = self.model.config.max_position_embeddings
        self.hparams.seqlen = self.model.config.max_position_embeddings
        self.triple_loss = TripletLoss(margin=float(init_args.margin))
        # number of training instances
        self.training_size = None
        # number of testing instances
        self.validation_size = None
        # number of test instances
        self.test_size = None
        # This is a dictionary to save the embeddings for source papers in test step.
        self.embedding_output = {}
        self.lossList = []

    def forward(self, input_ids, token_type_ids, attention_mask):
        # in lightning, forward defines the prediction/inference actions
        source_embedding = self.model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return source_embedding[1]

    """
    このメソッドでallennlp用のデータをロードして、トークナイズまで行う（tokentype_id, attention_maskなど)
    -> つまりこのメソッドに関わる箇所を書き換えればいい。
    """

    def _get_loader(self, split):
        path = "/workspace/dataserver/axcell/large/specter/" + \
            self.hparams.method + "/triple-" + \
            self.label + "-" + split + ".json"
        with open(path, 'r') as f:
            data = json.load(f)

        print(path)
        print("-----data length -----", len(data))
        path = "/workspace/dataserver/axcell/large/labeledAbst.json"
        with open(path, 'r') as f:
            labeledAbstDict = json.load(f)
        # 扱いやすいようにアブストだけでなくタイトルもvalueで参照できるようにしておく
        for title in labeledAbstDict:
            labeledAbstDict[title]["title"] = title

        dataset = MyData(data, labeledAbstDict, self.label, self.tokenizer)

        # pin_memory enables faster data transfer to CUDA-enabled GPU.
        loader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                            shuffle=False, pin_memory=False)

        # print(loader)
        return loader

    """
    これはよくわからん、絶対に呼び出されるやつ？
    """

    def setup(self, mode):
        self.train_loader = self._get_loader("train")

    """
    以下はすべてPytorch Lightningの指定のメソッド
    そのためこのファイル内には呼び出している箇所は無い。
    """
    """
    allennlp用のデータを読み取り、変換する
    """

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader_obj = self._get_loader('dev')
        return self.val_dataloader_obj

    def test_dataloader(self):
        return self._get_loader('test')

    """
    学習の設定等に関わるメソッド群
    """
    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(
            1, self.hparams.total_gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.batch_size * \
            self.hparams.grad_accum * num_devices
        # dataset_size = len(self.train_loader.dataset)
        """The size of the training data need to be coded with more accurate number"""
        dataset_size = training_size
        return (dataset_size / effective_batch_size) * self.hparams.num_epochs

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.lr, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        source_embedding = self.model(**batch[0])[1]
        pos_embedding = self.model(**batch[1])[1]
        neg_embedding = self.model(**batch[2])[1]

        loss = self.triple_loss(source_embedding, pos_embedding, neg_embedding)

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]

        self.log('train_loss', loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True)
        self.log('rate', lr_scheduler.get_last_lr()
                 [-1], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        self.lossList.append(loss.detach().cpu().numpy())
        # print(self.lossList)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        source_embedding = self.model(**batch[0])[1]
        pos_embedding = self.model(**batch[1])[1]
        neg_embedding = self.model(**batch[2])[1]

        loss = self.triple_loss(source_embedding, pos_embedding, neg_embedding)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'val_loss': loss}

    def _eval_end(self, outputs) -> tuple:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(
                avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
        results = {"avg_val_loss": avg_loss}
        for k, v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach().cpu().item()
        return results

    def validation_epoch_end(self, outputs: list) -> dict:
        ret = self._eval_end(outputs)

        self.log('avg_val_loss', ret["avg_val_loss"],
                 on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: list):
        # convert the dictionary of {id1:embedding1, id2:embedding2, ...} to a
        # list of dictionaries [{'id':'id1', 'embedding': 'embedding1'},{'id':'id2', 'embedding': 'embedding2'}, ...]
        embedding_output_list = [{'id': key, 'embedding': value.detach().cpu().numpy().tolist()}
                                 for key, value in self.embedding_output.items()]

        with open(self.hparams.save_dir+'/embedding_result.jsonl', 'w') as fp:
            fp.write('\n'.join(json.dumps(i) for i in embedding_output_list))

    def test_step(self, batch, batch_nb):
        source_embedding = self.model(**batch[0])[1]
        source_paper_id = batch[1]

        batch_embedding_output = dict(zip(source_paper_id, source_embedding))

        # .update() will automatically remove duplicates.
        self.embedding_output.update(batch_embedding_output)
        # return self.validation_step(batch, batch_nb)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=None,
                        help='path to the model (if not setting checkpoint)')
    parser.add_argument('--method')
    parser.add_argument('--margin', default=1)
    parser.add_argument('--label', default=None)
    parser.add_argument('--version', default=0)
    parser.add_argument('--input_dir', default=None,
                        help='optionally provide a directory of the data and train/test/dev files will be automatically detected')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--grad_accum', default=1, type=int)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--seed', default=1918, type=int)
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--limit_test_batches', default=1.0, type=float)
    parser.add_argument('--limit_val_batches', default=1.0, type=float)
    parser.add_argument('--val_check_interval', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", default=0.0,
                        type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="kwarg passed to DataLoader")
    parser.add_argument("--adafactor", action="store_true")
    parser.add_argument('--save_dir', required=True)

    parser.add_argument('--num_samples', default=None, type=int)
    parser.add_argument("--lr_scheduler",
                        default="linear",
                        choices=arg_to_scheduler_choices,
                        metavar=arg_to_scheduler_metavar,
                        type=str,
                        help="Learning rate scheduler")
    args = parser.parse_args()

    if args.input_dir is not None:
        files = glob.glob(args.input_dir + '/*')
        for f in files:
            fname = f.split('/')[-1]
            if 'train' in fname:
                args.train_file = f
            elif 'dev' in fname or 'val' in fname:
                args.dev_file = f
            elif 'test' in fname:
                args.test_file = f
    return args


def get_train_params(args):
    train_params = {}
    train_params["precision"] = 16 if args.fp16 else 32
    if (isinstance(args.gpus, int) and args.gpus > 1) or (isinstance(args.gpus, list) and len(args.gpus) > 1):
        train_params["distributed_backend"] = "ddp"
    else:
        train_params["distributed_backend"] = None
    train_params["accumulate_grad_batches"] = args.grad_accum
    train_params['track_grad_norm'] = -1
    train_params['limit_val_batches'] = args.limit_val_batches
    train_params['val_check_interval'] = args.val_check_interval
    train_params['gpus'] = args.gpus
    train_params['max_epochs'] = args.num_epochs
    train_params['log_every_n_steps'] = log_every_n_steps
    return train_params

# LINEに通知する関数


def line_notify(message):
    line_notify_token = 'Jou3ZkH4ajtSTaIWO3POoQvvCJQIdXFyYUaRKlZhHMI'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)


def main():
    try:
        args = parse_args()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.num_workers == 0:
            print("num_workers cannot be less than 1")
            return

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if ',' in args.gpus:
            args.gpus = list(map(int, args.gpus.split(',')))
            args.total_gpus = len(args.gpus)
        else:
            args.gpus = int(args.gpus)
            args.total_gpus = args.gpus

        if args.test_only:
            print('loading model...')
            model = Specter.load_from_checkpoint(args.test_checkpoint)
            trainer = pl.Trainer(
                gpus=args.gpus, limit_val_batches=args.limit_val_batches)
            trainer.test(model)

        else:
            if args.label:
                labelList = [args.label]
            else:
                labelList = ["title", "bg", "obj", "method", "res"]

            line_notify("172.21.64.47: specterのtrain.pyが開始")

            for label in labelList:
                args.label = label

                model = Specter(args)

                # default logger used by trainer
                logger = TensorBoardLogger(
                    save_dir=args.save_dir,
                    version=args.version,
                    name='pl-logs'
                )

                # second part of the path shouldn't be f-string
                dirPath = f'/workspace/dataserver/model_outputs/specter/{args.method}_{logger.version}/'
                filepath = dirPath + 'checkpoints/' + args.label + \
                    '-ep-{epoch}_avg_val_loss-{avg_val_loss:.3f}'
                checkpoint_callback = ModelCheckpoint(
                    filepath=filepath,
                    save_top_k=4,
                    verbose=True,
                    # monitors metrics logged by self.log.
                    monitor='avg_val_loss',
                    mode='min',
                    prefix=''
                )

                extra_train_params = get_train_params(args)

                trainer = pl.Trainer(logger=logger,
                                     checkpoint_callback=checkpoint_callback,
                                     **extra_train_params)

                trainer.fit(model)

                """
                ロスの可視化
                """
                dataPath = "/workspace/dataserver/axcell/large/specter/" + \
                    args.method + "/triple-" + \
                    args.label + "-train.json"
                with open(dataPath, 'r') as f:
                    data = json.load(f)

                fig = plt.figure()
                x = list(range(1, len(model.lossList) + 1))
                for i in range(1, args.num_epochs):
                    # trainningデータの長さをバッチサイズで割り、1を足す
                    vlineValue = (int(len(data)/args.batch_size)+1)*i
                    # print(vlineValue)
                    plt.vlines(x=vlineValue, ymin=0, ymax=2, colors="gray",
                               linestyles="dashed", label="epoch"+str(i))
                plt.legend()

                # ロスの線が潰れないように束でとって平均化する
                batch = 100
                pltLossList = []
                pltX = []
                for i in range(int(len(model.lossList)/batch)+1):
                    if i*batch+batch < len(model.lossList):
                        pltLossList.append(
                            np.mean(model.lossList[i*batch:i*batch+batch]))
                        print(i*batch, i*batch+batch)
                    else:
                        pltLossList.append(np.mean(model.lossList[i*batch:]))
                        print(i*batch)
                    pltX.append(i*batch)
                plt.plot(pltX, pltLossList)

                imgDirPath = dirPath + "image/"
                imgPath = imgDirPath + "loss-" + args.label + ".png"
                if not os.path.exists(imgDirPath):
                    os.mkdir(imgDirPath)
                fig.savefig(imgPath)

                with open(dirPath + "args.json", "w") as f:
                    json.dump(vars(args), f, indent=4)

                line_notify("172.21.65.47: specterのtrain.py" +
                            "の" + args.label + "の観点" + "が終了")

                del model
                del logger
                del trainer
                del filepath
                del fig
                torch.cuda.empty_cache()

    except Exception as e:
        print(traceback.format_exc())
        message = "172.21.65.47: Error: " + \
            os.path.basename(__file__) + " " + str(traceback.format_exc())
        line_notify(message)


if __name__ == '__main__':
    main()
