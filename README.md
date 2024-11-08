# KIOST-SST-Downscaling
===
[[Preprint](https://sstdv-project.github.io/template-project-page/static/pdfs/sample.pdf)]
[[Supplementary](https://sstdv-project.github.io/template-project-page/static/pdfs/sample.pdf)]
[[Project Page](https://sstdv-project.github.io/template-project-page/)]

## Project directory structure
```
.
├── README.md
├── LICENSE
├── code/
│   └── [Project source code]
└── data/
    └── [Synthetic data]
```

## Project guideline
Any public project SHOULD include:
* MIT License @ `LICENSE`
* Acknowledgement @ `README.md`
* BibTeX citation + link to PDF file @ `README`, if the project is accompanied with a research paper

Any public project SHOULD NOT include:
* Private data, undisclosed data, data with limited accessibility
  - Preferably, *any* data should be hosted outside of the repository.
* Personal information
  - *Unintended* personal information of researchers and developers within source code
  - Device IP address, password, secrets, file path, ...

Any Public project is encouraged to include:
* Project pages (GitHub pages or other platform)
* Examples as Colab/Jupyter notebook



## Dataset

| Product  | Available period | Resolution | Temp | Source | Agency |
| Temp | Temp | Spatial | Temporal | Temp | Temp |

* KAM Weather Data Service 'Open MET Data Portal'
* Ocean Observation - Sea Surface Temperature(SST) 
* https://data.kma.go.kr/data/sea/selectBuoyRltmList.do?pgmNo=52


## Requirements

First, install PyTorch meeting your environment (at least 1.7):
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Then, use the following command to install the rest of the libraries:
```bash
pip install tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg timm
```

## Features

- **Model:** Two models of LSTM and Transformer are applied to extreme value analysis. For the extreme value analysis, three methods of data transformation, Frechet and Gumbel extreme distribution loss (Zhang et al.,2021) are applied to two models.  
- **Time-series:** 16 time series of sea surface temperature (SST) around the waters of Korea Peninsula are used and the data can be freely access through KAM Weather Data Service 'Open MET Data Portal'. 
- **Gumbel Generalize Value Loss:** 4 cases of Gumbel distribution function according to the hyper parameter r are applied (r=1.0, 1.1, 1.5, 2.0). 
- **Freceht Generalize Value Loss:** 4 cases of Frechet distribution function according to the hyper parameters of a and s are applied. (a=10 s=1.7, a=13 s=1.7, a=15 s=1.7, a=15 s=2.0)


## Citation

```bibtex
@article{kim2023multi,
  title={Multi-source deep data fusion and super-resolution for downscaling sea surface temperature guided by Generative Adversarial Network-based spatiotemporal dependency learning},
  author={Kim, Jinah and Kim, Taekyung and Ryu, Joon-Gyu},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={119},
  pages={103312},
  year={2023},
  publisher={Elsevier}
}
```

## Acknowledgement

###### Korean acknowledgement
> 이 논문은 20__년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.00223446, 목적 맞춤형 합성데이터 생성 및 평가기술 개발)

###### English acknowledgement
> This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.00223446, Development of object-oriented synthetic data generation and evaluation methods)
