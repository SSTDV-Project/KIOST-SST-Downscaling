# KIOST-SST-Downscaling
===
[[Preprint](https://sstdv-project.github.io/template-project-page/static/pdfs/sample.pdf)]
[[Supplementary](https://sstdv-project.github.io/template-project-page/static/pdfs/sample.pdf)]
[[Project Page](https://sstdv-project.github.io/template-project-page/)]

![화면 캡처 2024-11-08 152601](https://github.com/user-attachments/assets/40393a73-2067-4971-b1a7-4349f8179b43)

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

![image04](https://github.com/user-attachments/assets/236073cf-669c-466c-9705-9db864a989e7)

![image03](https://github.com/user-attachments/assets/0f73eee1-764b-4e7f-a5b8-4bad77452a3c)

* ERA5 Download link : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab-overview *
* Please contact tkkim@kiost.ac.kr for other dataset

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

- **Model:** Two models of Generator and Discriminator are defined for adversarial learning. A generator synthesizing downscaled global/peninsula SST data is obtained by minimizing WGAN-GP loss (Gulrajani et al.,2017) and MSE loss between model output and insitu data.
- ![화면 캡처 2024-11-08 154414](https://github.com/user-attachments/assets/3bee283c-1948-4ccb-aa29-8c5a926cbbd9)
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
@article{gulrajani2017improved,
  title={Improved training of wasserstein gans},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron C},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## Acknowledgement

###### Korean acknowledgement
> 이 논문은 2023년-2026년 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임 (No.00223446, 목적 맞춤형 합성데이터 생성 및 평가기술 개발)

###### English acknowledgement
> This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.00223446, Development of object-oriented synthetic data generation and evaluation methods)
