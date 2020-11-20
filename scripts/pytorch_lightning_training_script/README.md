# SPECTER Training Script

`main.py` is a pytorch-lightning training script to train a SPECTER model from [`Scibert`]( https://huggingface.co/allenai/scibert_scivocab_cased). 

To train, please run the following code.

```bash
python main.py --save_dir path/to/save --gpus 1 --train_file path/to/training_set --dev_file path/to/val_set --test_file path/to/test_set --batch_size 2 --num_workers 0 --num_epochs 4 --grad_accum 256
```
e.g.,
```bash
python main.py --save_dir ./save --gpus 1 --train_file /mnt/nvme0n1p1/specterdata/train.pkl --dev_file /mnt/nvme0n1p1/specterdata/val.pkl --test_file /mnt/nvme0n1p1/specterdata/val.pkl --batch_size 2 --num_workers 0 --num_epochs 4 --grad_accum 256
```

To test, simply run the following code.

```bash
python main.py --save_dir path/to/save --gpus 1 --test_only $true --test_checkpoint path/to/model_checkpoint
```
e.g.,
```bash
python main.py --save_dir ./save --gpus 1 --test_only $true --test_checkpoint ./save/version_0/checkpoints/ep-epoch=3_loss-val_loss=0.206.ckpt
```

