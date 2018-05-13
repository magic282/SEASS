# SEASS
This repository contains code for the ACL 2017 paper "Selective Encoding for Abstractive Sentence Summarization"

## About this code

The experiments in the paper were done with an in-house deep learning tool. Therefore, we re-implement this as a reference.

**PyTorch version**: This code requires PyTorch v0.3.x.

**Python version**: This code requires Python3.


## How to run

### Prepare the dataset and code

You can download the processed data from [here](https://github.com/harvardnlp/sent-summary).
Or, you can process the dataset with [NAMAS](https://github.com/harvardnlp/NAMAS).

Make a folder for the code and data:
```bash
SEASS_HOME=~/workspace/seass
mkdir -p $SEASS_HOME/code
cd $SEASS_HOME/code
git clone --recursive https://github.com/magic282/SEASS.git
```
Put the data in the folder `$SEASS_HOME/code/data/giga` and organize them as:
```
seass
├── code
│   └── SEASS
│       └── seq2seq_pt
└── data
    └── giga
        ├── dev
        ├── models
        └── train
```
Since the validation set is large, you can sample a small set from it.

Collect vocabulary using `CollectVocab.py`.
Then put the vocab files in the `train` folder.

Modify `run.sh` according to your setting and files.
### Setup the environment
#### Package Requirements:
```
nltk scipy numpy pytorch
```
**Warning**: Older versions of NLTK have a bug in the PorterStemmer. Therefore, a fresh installation or update of NLTK is recommended.

A Docker image is also provided.
#### Docker image
```bash
docker pull magic282/pytorch:0.3.1
```
### Run training
The file `run.sh` is an example. Modify it according to your configuration.
#### Without Docker
```bash
bash $SEASS_HOME/code/SEASS/seq2seq_pt/run.sh $SEASS_HOME/data/giga $SEASS_HOME/code/SEASS/seq2seq_pt
```
#### With Docker
```bash
nvidia-docker run --rm -ti -v $SEASS_HOME:/workspace magic282/pytorch:0.3.1
```
Then inside the docker:
```bash
bash code/SEASS/seq2seq_pt/run.sh /workspace/data/giga /workspace/code/SEASS/seq2seq_pt
```
