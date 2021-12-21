# Empirical-Channel-CM

## An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems
This repository contains our implementation of the paper, "An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems".
[[Paper link](https://www.isca-speech.org/archive/interspeech_2021/zhang21ea_interspeech.html)] [[arXiv](https://arxiv.org/pdf/2104.01320.pdf)] [[Video presentation](https://www.youtube.com/watch?v=t6qtehKer6w)] 

### Cross-Dataset Studies
Existing datasets:
[ASVspoof2019LA](https://datashare.ed.ac.uk/handle/10283/3336),
[ASVspoof2015](https://datashare.ed.ac.uk/handle/10283/853),
[VCC2020 training data](https://zenodo.org/record/4345689#.YVp3UlNKgt0),
[VCC2020 submissions](https://zenodo.org/record/4433173)


Augmented data:

Training + Development: [ASVspoof2019LA-Sim v1.0](https://zenodo.org/record/5548622) 

Evaluation: [ASVspoof2019LA-Sim v1.1](https://zenodo.org/record/5794671)

###  Channel Robust Strategies

#### Run the training code
```
python3 train.py -o /path/to/output/the/model
```
The options:

--AUG use the plain augmentation

--MT_AUG use the multitask augmentation

--ADV_AUG use the adversarial augmentation

#### Run the testing code
```
python3 test.py -m /path/to/the/trained/model --task ASVsppof2019LA
```
The options for testing on different dataset:

ASVspoof2019LA, ASVspoof2015, VCC2020, ASVspoof2019LASim

The code is based on our previous work "One-class Learning Towards Synthetic Voice Spoofing Detection" [[code link](https://github.com/yzyouzhang/AIR-ASVspoof)] [[paper link](https://ieeexplore.ieee.org/document/9417604)]


### Citation
```
@inproceedings{zhang21ea_interspeech,
  author={You Zhang and Ge Zhu and Fei Jiang and Zhiyao Duan},
  title={{An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={4309--4313},
  doi={10.21437/Interspeech.2021-1820}
}
```

Please also feel free to check out our follow-up work:

[1] Chen, X., Zhang, Y., Zhu, G., Duan, Z. (2021) UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021. Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge, 75-82, doi: 10.21437/ASVSPOOF.2021-12 [[link](https://www.isca-speech.org/archive/pdfs/asvspoof_2021/chen21_asvspoof.pdf)] [[code](https://github.com/yzyouzhang/ASVspoof2021_AIR)]
