# Compress to Focus: Efficient Coordinate Compression for Policy Optimization in Multi-Turn GUI Agents

<!-- 徽章区域：显得专业且信息丰富 -->
[![Project Page](https://img.shields.io/badge/Project-Page-Green.svg)](https://user.github.io/CCPO/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2601.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Pytorch 2.7+](https://img.shields.io/badge/pytorch-2.7+-green.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-goldenrod)](https://huggingface.co/)

<!-- **[Your Name]**, **[Co-author Name]**, ... -->
**Yurun Song\*, Jiong Yin\*, Rongjunchen Zhang, Ian Harris**

<!-- **[Institution Name]** -->



## 🔥 News
- **[2026-01-12]** 🚀 Code and pre-trained models are released!
- **[2026-01-XX]** 📄 Our paper "[Paper Title]" is now available on [arXiv](https://arxiv.org/abs/2601.xxxxx).


## 🚀 Introduction

The official implementation of the paper "Compress to Focus: Efficient Coordinate Compression for Policy Optimization in Multi-Turn GUI Agents". 

> **Abstract:** *Multi-turn GUI agents enable complex task completion through sequential decision-making, but suffer from severe context inflation as interaction history accumulates. Existing strategies either sacrifice long-term context via truncation or compromise spatial structure through token pruning. In this paper, we propose Coordinate Compression Policy Optimization (CCPO), an efficient policy optimization framework that couples visual compression with policy optimization for multi-turn GUI agents. CCPO introduces Coordinate-Aware Spatial Compression (CASC), which aggregates coordinates from multiple rollouts to capture target-relevant regions and progressively narrow historical attention around key visual areas. From interactions across rollouts, CASC adaptively constructs attention boundaries that concentrate computation on the most informative regions of the scene. We further design a Distance-Based Advantage that provides fine-grained learning signals based on distance rather than binary correctness, improving both grounding accuracy and compression quality. Extensive experiments demonstrate that CCPO achieves SOTA performance across four benchmarks with up to 55% token compression and 3.8$\times$ training speedup.*

<!-- 这里放一张方法流程图，这是顶会项目的标配 -->
![Method Overview](assets\framework-12-28_1.png)
*Figure 1: Overview of the CCPO framework.*

### ✨ Key Features
- **SOTA Performance**: Achieves state-of-the-art results on [Benchmark Name].
- **Efficiency**: 2x faster training compared to [Baseline Method].
- **Easy-to-use**: Simple API and comprehensive documentation.






## 🛠️ Installation

### Requirements
- Linux
- Python 3.12+
- PyTorch 2.7+
- CUDA 12.8+
- Please refer to [requirements.txt](requirements.txt) for other dependencies.

### Setup
```bash
# Clone the repository
git clone https://github.com/username/CCPO.git
cd CCPO

# Create a conda environment
conda create -n ccpo python=3.9
conda activate ccpo

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```



## 📂 Data Preparation

We evaluate CCPO on four major benchmarks: **Android Control**, **GUI Odyssey**, **Mind2Web**, and **AITW**.

Please download the datasets from their official websites and run the provided scripts to convert them into the required format.

| Dataset | Download Link | Preprocessing Script |
| :--- | :---: | :--- |
| **Android Control** | [Link]([INSERT_URL_HERE]) | `python [INSERT_SCRIPT_PATH_HERE] ...` |
| **GUI Odyssey** | [Link]([INSERT_URL_HERE]) | `python [INSERT_SCRIPT_PATH_HERE] ...` |
| **Mind2Web** | [Link]([INSERT_URL_HERE]) | `python [INSERT_SCRIPT_PATH_HERE] ...` |
| **AITW** | [Link]([INSERT_URL_HERE]) | `python [INSERT_SCRIPT_PATH_HERE] ...` |

After preprocessing, please organize the data as follows:

```
data/
├── android_control/
├── gui_odyssey/
├── mind2web/
└── aitw/
```

---

## 🏃 Usage

### 1. Training
To train the CCPO model with 8 GPUs:

```bash
bash scripts/train_ccpo.sh
```

### 2. Evaluation
To evaluate the pre-trained model:

```bash
python evaluation_gui_o_sequence_new_version.py \
    --save_path path/to/save/results \
    --model_path path/to/model \
    --his_num 4
```

---

## 📊 Model Zoo

We provide pre-trained models (3B and 7B) for reproduction.

| Dataset | CCPO-3B | CCPO-7B |
| :--- | :---: | :---: |
| **Android Control** | [Download](#) | [Download](#) |
| **GUI Odyssey** | [Download](#) | [Download](#) |
| **Mind2Web** | [Download](#) | [Download](#) |
| **AITW** | [Download](#) | [Download](#) |

---

## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{author2026ccpo,
  title={xxxx},
  author={xxxx},
  booktitle={xxxx},
  year={2026}
}
```

## 🙏 Acknowledgement

This project is built upon [UI-S1](https://github.com/X-PLUG/MobileAgent/tree/main/UI-S1), [SimpAgent](https://github.com/JiuTian-VL/SimpAgent), and [verl-agent](https://github.com/langfengQ/verl-agent). We thank the authors for their great code.
