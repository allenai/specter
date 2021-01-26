# SPECTER: Document-level Representation Learning using Citation-informed Transformers

[**SPECTER**](#specter-document-level-representation-learning-using-citation-informed-transformers) | [**Pretrained models**](#How-to-use-the-pretrained-model) | [**Training your own model**](#advanced-training-your-own-model) | 
[**SciDocs**](https://github.com/allenai/scidocs) | [**Public API**](#Public-api) | 
[**Paper**](https://arxiv.org/pdf/2004.07180.pdf) | [**Citing**](#Citation) 


This repository contains code, link to pretrained models, instructions to use [SPECTER](https://arxiv.org/pdf/2004.07180.pdf) and link to the [SciDocs](https://github.com/allenai/scidocs) evaluation framework.


***** New Jan 2021: HuggingFace models *****

Specter is now accessible through HuggingFace's transformers library.  

*Thanks to [@zhipenghoustat](https://github.com/zhipenghoustat) for providing the Huggingface training scripts and the checkpoint.*

See below:

# How to use the pretrained model

## 1- Through Huggingface Transformers Library

Requirement: `pip install --upgrade transformers==4.2`

```python
from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
          {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# concatenate title and abstract
title_abs = [d['title'] + ' ' + (d.get('abstract') or '') for d in papers]
# preprocess the input
inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
result = model(**inputs)
# take the first token in the batch as the embedding
embeddings = result.last_hidden_state[:, 0, :]
```

A sample script to run the model in batch mode on a dataset of papers is provided under `scripts/embed_papers_hf.py`

How to use:
```
CUDA_VISIBLE_DEVICES=0 python scripts/embed_papers_hf.py \
--data-path path/to/paper-metadata.json \
--output path/to/write/output.json \
--batch-size 8
```

** Note that huggingface model yields slightly higher average results than those reported in the paper.
To reproduce our exact numbers use our original implementation see [**reproducing results**](#How-to-reproduce-our-results).

*Expected SciDocs results from the huggingface model:*

| mag-f1 	| mesh-f1 	| co-view-map 	| co-view-ndcg 	| co-read-map 	| co-read-ndcg 	| cite-map 	| cite-ndcg 	| cocite-map 	| cocite-ndcg 	| recomm-ndcg 	| recomm-P@1 	| Avg  	|
|--------	|---------	|-------------	|--------------	|-------------	|--------------	|----------	|-----------	|------------	|-------------	|-------------	|------------	|------	|
| 79.4   	| 87.7    	| 83.4        	| 91.4         	| 85.1        	| 92.7         	| 92.0     	| 96.6      	| 88.0       	| 94.7        	| 54.6        	| 20.9       	| 80.5 	|

## 2- Through this repo

1 - Clone the repo and download the pretrained model and supporting files:

### Download

Download the tar file at: [**download**](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz) [833 MiB]  
The compressed archive includes a `model.tar.gz` file which is the pretrained model as well as supporting files that are inside a `data/` directory. 

Here are the commands to run:

```python
git clone git@github.com:allenai/specter.git

cd specter

wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz

tar -xzvf archive.tar.gz 
```


2 - Install the environment:

```python
conda create --name specter python=3.7 setuptools  

conda activate specter  

# if you don't have gpus, remove cudatoolkit argument
conda install pytorch cudatoolkit=10.1 -c pytorch   

pip install -r requirements.txt  

python setup.py install
```


3 - Embed papers or documents using SPECTER

Specter requires two main files as input to embed the document. A text file with ids of the documents you want to embed and a json metadata file consisting of the title and abstract information. Sample files are provided in the `data/` directory to get you started. Input data format is according to:

```python
metadata.json format:

{
    'doc_id': {'title': 'representation learning of scientific documents',
               'abstract': 'we propose a new model for representing abstracts'},
}
```

To use SPECTER to embed your data use the following command:

```python
python scripts/embed.py \
--ids data/sample.ids --metadata data/sample-metadata.json \
--model ./model.tar.gz \
--output-file output.jsonl \
--vocab-dir data/vocab/ \
--batch-size 16 \
--cuda-device -1
```

Change `--cuda-device` to `0` or your specified GPU if you want faster inference.  
The model will run inference on the provided input and writes the output to `--output-file` directory (in the above example `output.jsonl` ).  
This is a jsonlines file where each line is a key, value pair consisting the id of the embedded document and its specter representation.


# Public API

A collection of public APIs for embedding scientific papers using Specter is available at: [**allenai/paper-embedding-public-apis**](https://github.com/allenai/paper-embedding-public-apis) 


# How to reproduce our results

In order to reproduce our results please refer to the [SciDocs](https://github.com/allenai/scidocs) repo where we provide the embeddings for the evaluation tasks and instructions on how to run the benchmark to get the results.

# Advanced: Training your own model

First follow steps 1 and 2 from the [Pretrained models](#How-to-use-the-pretrained-model) section to download the supporting files and install the environment.

Next you need to create pickled training instances using the `specter/data_utils/create_training_files.py` script and then use the resulting files as input to the `scripts/run-exp-simple.sh` script.  

You will need the following files:
* `data.json` containing the document ids and their relationship.  
* `metadata.json` containing mapping of document ids to textual fiels (e.g., `title`, `abstract`)
* `train.txt`,`val.txt`, `test.txt` containing document ids corresponding to train/val/test sets (one doc id per line).

The `data.json` file should have the following structure (a nested dict):  
```python
{"docid1" : {  "docid11": {"count": 1}, 
               "docid12": {"count": 5},
               "docid13": {"count": 1}, ....
            }
"docid2":   {  "docid21": {"count": 1}, ....
....}
```

Where `docids` are ids of documents in your data and `count` is a measure of importance of the relationship between two documents. In our dataset we used citations as indicator of relationship where `count=5` means direct citation while `count=1` refers to a citation of a citation.  
  
The `create_training_files.py` script processes this structure with a triplet sampler that selects both easy and hard negatives (as described in the paper) according the `count` value in the above structure. For example papers with `count=5` are considered positive candidates, papers with `count=1` considered hard negatives and other papers that are not cited are easy negatives. You can control the number of hard negatives by setting `--ratio_hard_negatives` argument in the script.  

- Create preprocessed training files:  
```python
python specter/data_utils/create_training_files.py \
--data-dir data/training \
--metadata data/training/metadata.json \
--outdir data/preprocessed/
```

After preprocessing the data you will have three pickled files containing training instannces as well as a `metrics.json` showing number of examples in each set. Use the following script to start training the model:

- Run the training script
```python
./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet \
-s model-output/ --num-epochs 2 --batch-size 4 \
--train-path data/preprocessed/data-train.p --dev-path data/preprocessed/data-val.p \
--num-train-instances 55 --cuda-device -1
```

In this example: The model's checkpoint and logs will be stored in `model-output/ `.  
Note that you need to set the correct `--num-train-instances` for your dataset. This number is stored in `metrics.json` file output from the preprocessing step.
You can monitor the training progress using `tensorboard`:  
`tensorboard --logdir model-output/  --bind_all`

### 

# SciDocs benchmark

SciDocs evaluation framework consists of a suite of evaluation tasks designed for document-level tasks.

Link to SciDocs: 

*   [https://github.com/allenai/scidocs](https://github.com/allenai/scidocs)


# Citation

Please cite the [SPECTER paper](https://arxiv.org/pdf/2004.07180.pdf) as:  

```bibtex
@inproceedings{specter2020cohan,
  title={{SPECTER: Document-level Representation Learning using Citation-informed Transformers}},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}
```
