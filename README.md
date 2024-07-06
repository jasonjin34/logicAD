# LogicalAD

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="assets/github_repo.png" alt="Logo" width="600" height="250">
  <h3 align="center">Logical Anomaly Detection</h3>
</div>

This is the official Logicial Anomaly Detection Algorithm
developed by [Jin Er*](er.jin@lfb.rwth-aachen.de), Qihui Feng, Yongli Mou

## Table of contents

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#run-training">Run Training</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->
## Getting Started

We recommend to use virtual environment for setting environment. This package have tested with `python==3.10.13` under `ubuntu 22.04.4 LTS`

## Modify dot files 
Please check the modify `.env` file and change it to the location where you save your package

### Prerequisites
Please make sure you have `miniconda` or `anaconda` installed in your system

### Installation

_Below is an example of how you can install all the relevant packages_

1. Create virual environment
   ```sh
    # create env with miniconda/anaconda
    yes | conda create -n logic python=3.10
	pip install torchvision==0.12.0+cu113 torch==1.11.0+cu113 -i https://download.pytorch.org/whl/cu113
    pip install -e .
    pip install requirements.txt
   ```
# Source Packages

## Running / Training
The config file is saved in `src/anomalib/models/logicad/config.yaml`
```bash
python tools/train.py --config ./src/anomalib/models/logicad/config.yaml
```
or
```bash
python tools/train.py --model logicad
```

## Acknowledges 
This package is built based on anomalib, openclip, lighting and hydra

```tex
@misc{falcon2019pytorch,
  title={PyTorch Lightning The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate},
  author={Falcon, W and Team, TPL},
  year={2019}
}

@Misc{Yadan2019Hydra,
  author =       {Omry Yadan},
  title =        {Hydra - A framework for elegantly configuring complex applications},
  howpublished = {Github},
  year =         {2019},
  url =          {https://github.com/facebookresearch/hydra}
}
@misc{anomalib,
      title={Anomalib: A Deep Learning Library for Anomaly Detection},
      author={Samet Akcay and
              Dick Ameln and
              Ashwin Vaidya and
              Barath Lakshmanan and
              Nilesh Ahuja and
              Utku Genc},
      year={2022},
      eprint={2202.08341},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
