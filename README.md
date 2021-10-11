# Empirical-Channel-CM

## An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems

### Cross-Dataset Studies
Existing datasets:
[ASVspoof2019LA](https://datashare.ed.ac.uk/handle/10283/3336),
[ASVspoof2015](https://datashare.ed.ac.uk/handle/10283/853),
[VCC2020 training data](https://zenodo.org/record/4345689#.YVp3UlNKgt0),
[VCC2020 submissions](https://zenodo.org/record/4433173)


Augmented data:
[ASVspoof2019LA-Sim](https://zenodo.org/record/5548622)

###  Channel Robust Strategies
Run the training code
```
python3 train.py -o /path/to/output/the/model
```
The options:

--AUG use the plain augmentation

--MT_AUG use the multitask augmentation

--ADV_AUG use the adversarial augmentation

The code is based on our previous work "One-class Learning Towards Synthetic Voice Spoofing Detection" [[code link](https://github.com/yzyouzhang/AIR-ASVspoof)]


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

[1] Chen, X., Zhang, Y., Zhu, G., Duan, Z. (2021) UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021. Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge, 75-82, doi: 10.21437/ASVSPOOF.2021-12 [[link](https://www.isca-speech.org/archive/pdfs/asvspoof_2021/chen21_asvspoof.pdf)]
