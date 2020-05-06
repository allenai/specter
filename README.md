# SPECTER: Document-level Representation Learning using Citation-informed Transformers

[**SPECTER**](#specter-document-level-representation-learning-using-citation-informed-transformers) | [**Pretrained models**](#How-to-use-the-pretrained-model) | 
[**SciDocs**](https://github.com/allenai/scidocs) | [**Public API**](#Public-api) | 
[**Paper**](https://arxiv.org/pdf/2004.07180.pdf) | [**Citing**](#Citation) 


This repository contains code, link to pretrained models, instructions to use [SPECTER](https://arxiv.org/pdf/2004.07180.pdf) and link to the [SciDocs](https://github.com/allenai/scidocs) evaluation framework.

# How to use the pretrained model

1 - Clone the repo and download the pretrained model and supporting files:

### Download

Download the tar file at: [**download**](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz) [833 MiB]  
The compressed archive includes a `model.tar.gz` file which is the pretrained model as well as supporting files that are inside a `data/` directory. 

Here are the commands to run:

```ruby
git clone git@github.com:allenai/specter-internal.git

cd specter-internal

wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz

tar -xzvf https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz 
```


2 - Install the environment:

```ruby
conda create --name specter python=3.7 setuptools  

conda activate specter  

# if you don't have gpus, remove cudatoolkit argument
conda install pytorch cudatoolkit=10.1 -c pytorch   

pip install -r requirements.txt  
```

3 - Embed papers or documents using SPECTER

Specter requires two main files as input to embed the document. A text file with ids of the documents you want to embed and a json metadata file consisting of the title and abstract information. Sample files are provided in the `data/` directory to get you started. Input data format is according to:

```ruby
metadata.json format:

{
    'doc_id': {'title': 'representation learning of scientific documents',
               'abstract': 'we propose a new model for representing abstracts'},
}
```

To use SPECTER to embed your data use the following command:

```ruby
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

# Training your own model

Instructions on training SPECTER on your own data is available on request.

# SciDocs benchmark

SciDocs evaluation framework consists of a suite of evaluation tasks designed for document-level tasks.

Link to SciDocs: 

*   [https://github.com/allenai/scidocs](https://github.com/allenai/scidocs)


# Citation

Please cite the [SPECTER paper](https://arxiv.org/pdf/2004.07180.pdf) as:  

```ruby
@inproceedings{specter2020cohan,
  title={SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  author={Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle={ACL},
  year={2020}
}
```
