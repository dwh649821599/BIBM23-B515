# BIBM23-B515
This repository contains the experimental code for IEEE BIBM 2023 paper "[What will regularized continuous learning performs if it was used to medical image segmentation: a preliminary analysis](https://ieeexplore.ieee.org/abstract/document/10385386)"

## Usage
### Code
This directory contains the source code for 8 different continuous learning methods on medical image segmentation tasks.
### Data
This directory contains the dataset used in the experiment and the corresponding code to make a PyTorch Dataset. The dataset is not included in the repository. Please download the dataset from the original source and place it in the corresponding location in the `data` directory.
### Model
This directory contains the code of U-Net model and its variants used in the experiment.

## Requirements
The code is tested with the following configuration:
- Python 3.9
- PyTorch 1.7.1
- CUDA 10.1
- Ubuntu 22.04

## Dataset
The link to the dataset used in the experiment is as follows. Place it in the corresponding location in the `data` directory:
### Cardiac
- M&Ms: https://www.ub.edu/mnms/
- ACDC: https://www.creatis.insa-lyon.fr/Challenge/acdc/
- SCD: https://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/
- LVSC: https://www.cardiacatlas.org/lv-segmentation-challenge/
### Fundus
- CHASE: https://blogs.kingston.ac.uk/retinal/chasedb1/
- DRHAGIS: https://medicine.uiowa.edu/eye/rite-dataset
- RITE: https://medicine.uiowa.edu/eye/rite-dataset
- STARE: https://cecas.clemson.edu/~ahoover/stare/

## Citation
If you use parts of the source code in your research, please cite this publication:
```
@inproceedings{dai2023will,
  title={What will regularized continuous learning performs if it was used to medical image segmentation: a preliminary analysis},
  author={Dai, Weihao and Feng, Chaolu and Chen, Shuaizheng and Li, Wei and Yang, Jinzhu and Zhao, Dazhe},
  booktitle={2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={1860--1863},
  year={2023},
  organization={IEEE}
}
```
