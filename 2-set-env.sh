#!/bin/bash
set -x
conda create --name specter python=3.7 setuptools
conda activate specter  

# if you don't have gpus, remove cudatoolkit argument
conda install pytorch cudatoolkit=10.1 -c pytorch   

pip install -r requirements.txt  

python setup.py install