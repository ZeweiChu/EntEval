# Ent Eval

This repository contains the code for EntEval
[EntEval: A Holistic Evaluation Benchmark for Entity Representations](https://arxiv.org/abs/1909.00137) (EMNLP 2019).

The structure of this repo:
- ```enteval```: the Entity Evaluation framework

Pretrained ELMo baseline model and ELMo hyperlink model can be downloaded from [https://drive.google.com/file/d/1hwE2Po_mmypgc3QsrpBRXGr3nyT1WIka/view?usp=sharing](https://drive.google.com/file/d/1hwE2Po_mmypgc3QsrpBRXGr3nyT1WIka/view?usp=sharing)

The EntEval evaluation dataset can be downloaded from [https://drive.google.com/file/d/1oEWXb7u81JaYQFycVTxow5anhWEG4KFz/view?usp=sharing](https://drive.google.com/file/d/1oEWXb7u81JaYQFycVTxow5anhWEG4KFz/view?usp=sharing), please download it and untar it in the EntEval main directory. 

Evaluation example code (You will need to set the elmo path)
```
example/eval_elmo.py
```
Note that in this evaluation script, we evaluate per layer ELMo and averaged ELMo layers, different from the paper impelementation. In the paper, the weights of each layer are trained as model parameters. 

The code is tested under the following environment/versions:
- Python 3.6.2
- PyTorch 1.0.0
- numpy 1.16.0

Some code in this repo is adopted from [SentEval](https://github.com/facebookresearch/SentEval). 

#### Pretraining

The pretraining WikiEnt data can be downloaded from 
[https://drive.google.com/drive/folders/1q3csyFdSQNiN6dMK19ahrmHQwijBAY38?usp=sharing](https://drive.google.com/drive/folders/1q3csyFdSQNiN6dMK19ahrmHQwijBAY38?usp=sharing)
v1 is the version we used in the paper, v2 is the updated version


## Reference

```
@inproceedings{mchen-enteval-19,
  author    = {Mingda Chen and Zewei Chu and Yang Chen and Karl Stratos and Kevin Gimpel},
  title     = {EntEval: A Holistic Evaluation Benchmark for Entity Representations},
  booktitle = {Proc. of {EMNLP}},
  year      = {2019}
}
```

