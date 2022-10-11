[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.10-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eth-nlped/scaffolding-generation/blob/main/LICENSE)

# Automatic Generation of Scaffolding Questions for Learning Math
This repository contains code of the paper:

[Automatic Generation of Scaffolding Questions for Learning Math]() (EMNLP 2022).  
_Kumar Shridhar*, Jakub Macina*, Mennatallah El-Assady, Tanmay Sinha, Manu Kapur and Mrinmaya Sachan_

## Dataset
[GSM8K Dataset](https://github.com/openai/grade-school-math)

## Training
```
python train.py train
```

## Running inference
```
python sample.py test
```
