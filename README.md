# Introduction

This project is created mainly to understand the transformer model of 
Neural Machine Translation. The most modules of the code has been taken from- 
http://nlp.seas.harvard.edu/2018/04/03/attention.html


## Pre-requisites
* python >= 3.6
* pytorch >= 0.5
* numpy 
* torchtext
* spacy (to train the de-en model)

## Quick Start
Train toy model on synthesized data.
```bash
$python train_toy.py
```
or

Train a real de-en model using default configuration. To download the prepared data
use the following spacy command.

```bash
$python -m spacy download en
$python -m spacy download de
```

```bash
$python train.py
```
