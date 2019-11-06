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

## Experiments
Our experiment results are as follows:
||CAP                          |CERP  |EFP                                          |ET  |ESR |ER  |ED  |Average|
|------|-----------------------------|------|---------------------------------------------|----|----|----|----|-------|
|GloVe |71.9                         |52.6  |67.0                                         |10.3|50.9|40.8|41.2|47.8   |
|BERT Base mix|80.6                         |65.6  |74.8                                         |32.0|28.8|42.2|50.6|53.5   |
|BERT Large mix|79.1                         |66.9  |76.7                                         |32.3|32.6|48.8|54.3|55.8   |
|5.5B ELMo|80.2                         |61.2  |75.8                                         |35.6|60.3|46.8|51.6|58.8   |
|Hyperlinking ELMo baseline|78.0                         |59.6  |71.5                                         |31.3|61.6|46.5|48.5|56.7   |
|Hyperlinking ELMo|76.9                         |59.9  |72.4                                         |32.2|59.7|45.7|49.0|56.5   |
|Hyperlinking ELMo without context|73.5                         |59.4  |71.1                                         |33.2|53.3|44.6|48.9|54.9   |
|Hyperlinking ELMo with entity mention|76.2                         |60.4  |70.9                                         |33.6|49.0|42.9|49.3|54.6   |


## Reference

```
@inproceedings{mchen-enteval-19,
  author    = {Mingda Chen and Zewei Chu and Yang Chen and Karl Stratos and Kevin Gimpel},
  title     = {EntEval: A Holistic Evaluation Benchmark for Entity Representations},
  booktitle = {Proc. of {EMNLP}},
  year      = {2019}
}
```

