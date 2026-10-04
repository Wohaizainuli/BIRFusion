<h1 align="center">BRIFusion</h1>

<p align="center">
  <strong>Bio-Inspired Response Enhancement and Reliability-Guided Routing<br>
  for Robust Multimodal Image Fusion</strong>
</p>

> 📌 **Release Notice**
>
> This repository currently provides an **illustrative implementation** to help researchers understand the main ideas and workflow of BRIFusion and facilitate academic discussion.
>
> **Pretrained model weights are not provided at this stage. The complete codebase will be released after the paper is accepted.**
>
> The current demonstration code introduces the method and its training pipeline; the complete experimental implementation will be included in the subsequent release.

## 🔍 Overview

BRIFusion addresses infrared and visible image fusion under degraded imaging conditions, including low illumination, noise, and reduced contrast. It combines local feature enhancement with reliability-guided information selection to preserve complementary thermal cues and scene details.

The method comprises three main designs:

- **Degradation-Conditioned Bio-Inspired Response Enhancement (DC-BRE):** adapts nonlinear feature responses to local image conditions using condition-dependent Hill responses and spatial response mixing.
- **Reliability-Guided Complementary Routing (RGCR):** combines modality reliability and content cues to determine fusion weights and select complementary experts within local windows.
- **Local Degradation Intervention Consistency (LDIC):** supervises reliability changes under local degradation and encourages stable fusion outputs outside the intervention neighborhood.

### Overall Architecture

Aligned infrared and visible images are processed by a shared-parameter Stem and four encoding stages. DC-BRE enhances features at the first two scales, while reliability prediction and RGCR guide multiscale fusion. The Fusion Decoder reconstructs the fused luminance, which is combined with the original visible chrominance for color output.

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/Fig2.jpg?raw=true" alt="Overall architecture of BRIFusion" width="100%">
</p>
<p align="center">
  <em>Fig. 1. Overall architecture of BRIFusion, including condition encoding, multiscale feature extraction, response enhancement, reliability-guided fusion, and auxiliary restoration branches.</em>
</p>

### Bio-Inspired Response Enhancement

DC-BRE receives both encoder features and modality-specific condition features. Local parameter modulation adjusts the Hill responses, spatial weighting combines their outputs, and multiscale refinement produces enhanced features through a residual connection.

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/Fig3.jpg?raw=true" alt="Structure of the DC-BRE module" width="100%">
</p>
<p align="center">
  <em>Fig. 2. Structure of DC-BRE: conditional response generation, spatial response mixing, and multiscale refinement.</em>
</p>

### Reliability-Guided Routing

At each scale, Rhead predicts reliability maps from modality features and their corresponding condition features. Joint Weighting determines the contributions of the two modalities, and window-wise Top-2 routing selects complementary experts for residual refinement.

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/Fig4.jpg?raw=true" alt="Reliability prediction and RGCR" width="100%">
</p>
<p align="center">
  <em>Fig. 3. Reliability prediction and RGCR. Reliability maps guide modality weighting and local expert selection, while mean reliability controls the contribution of expert refinement.</em>
</p>

The main network is implemented in `model/brifusion.py`, and the enhancement and routing modules are defined in `model/bri_layers.py`.

## 📦 Datasets

The fusion experiments use three public datasets:

| Dataset | Role in this study | Official source |
| :--- | :--- | :--- |
| **LLVIP** | Infrared–visible fusion in low-light scenes | [Dataset website](https://bupt-ai-cz.github.io/LLVIP/) |
| **M3FD** | Fusion evaluation and downstream object detection | [TarDAL / M3FD](https://github.com/dlut-dimt/TarDAL) |
| **MSRS** | Infrared–visible fusion in road scenes | [Dataset repository](https://github.com/Linfeng-Tang/MSRS) |

Download the datasets from their official sources and follow the corresponding access and usage instructions. The M3FD inputs used for the degradation experiments are additionally processed with brightness attenuation, blur, and additive noise.

### Data Organization

For supervised training with paired degraded inputs and clean references, organize the data as follows:

```text
data/
├── train/
│   ├── Vis/       # Visible input images
│   ├── Inf/       # Infrared input images
│   ├── Vis_gt/    # Clean visible reference images
│   └── Inf_gt/    # Clean infrared reference images
└── test/
    ├── Vis/       # Visible test images
    └── Inf/       # Infrared test images
```

Corresponding images must share the same filename stem, have matching spatial dimensions, and be spatially aligned. Prepare the degraded inputs and reference pairs before using `--mode paired`; the public datasets do not necessarily provide this four-folder layout directly. Keep training, validation, and test scenes separate.

For online degradation synthesis from clean paired images, use `--mode synthetic`. The loader uses `Vis_gt/Inf_gt` when available and otherwise uses `Vis/Inf` as the clean source pair.

## ⚙️ Installation

The current code uses **Python 3.10+** and **PyTorch 2.3+**. Install a PyTorch build compatible with your device, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. Replace the example data and output paths with your own directories.

## 🚀 Model Training

BRIFusion uses two training stages. The default settings are **100 epochs per stage**, a **batch size of 4**, and **256 × 256 training crops**. Loss weights and model options are defined in `configs/brifusion.json`.

### Stage 1: Restoration and Reliability Pretraining

Train the feature extraction branches, response enhancement modules, reliability heads, and auxiliary Restoration Decoders:

```bash
python train_main0.py --data data/train --mode paired --output runs/stage0 --device cuda:0 --amp
```

### Stage 2: Joint Fusion Training

Initialize from the first-stage checkpoint and jointly optimize the fusion framework:

```bash
python train_main1.py --data data/train --mode paired --init runs/stage0/last.pth --output runs/stage1 --device cuda:0 --amp
```

Training combines restoration, fusion, reliability, expert load-balancing, and LDIC objectives. Their implementations are provided in `utils/bri_losses.py` and `utils/degradations.py`.

- Add `--val-data data/val` to use a separate validation set with the same folder structure as the training set.
- `last.pth` is saved after each epoch; `best.pth` is also saved when a validation set is provided.
- Training settings and loss records are written to `config.json` and `metrics.jsonl` in the output directory.
- For CPU execution, replace `--device cuda:0` with `--device cpu` and omit `--amp`.

## 🧪 Model Testing

After training, run inference with a fusion-stage checkpoint:

```bash
python infer_brifusion.py --checkpoint runs/stage1/last.pth --vi data/test/Vis --ir data/test/Inf --output results/BRIFusion --device cuda:0 --save-quality
```

Fused images are saved as PNG files in `results/BRIFusion/`. The network processes visible luminance and infrared intensity; color outputs reuse the visible chrominance. Add `--grayscale` to save grayscale fusion results.

The `--save-quality` option exports reliability visualizations to the `quality/` subfolder. Inference preserves the original image dimensions through internal padding and cropping. Parameter counts and model-forward timing are recorded in `timing.json`.

## 🖼️ Fusion Results

The following figures present qualitative comparisons from the manuscript. Red boxes identify regions shown in the enlarged views.

### LLVIP

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/c1.jpg?raw=true" alt="Qualitative fusion comparison on LLVIP" width="100%">
</p>
<p align="center">
  <em>Fig. 4. Qualitative comparison on LLVIP, illustrating nighttime scene visibility, target saliency, and local vegetation detail.</em>
</p>

### M3FD

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/c2.jpg?raw=true" alt="Qualitative fusion comparison on M3FD" width="100%">
</p>
<p align="center">
  <em>Fig. 5. Qualitative comparison on M3FD under degraded conditions, highlighting structural contrast and the visibility of roadside vegetation and scene boundaries.</em>
</p>

### MSRS

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/c3.jpg?raw=true" alt="Qualitative fusion comparison on MSRS" width="100%">
</p>
<p align="center">
  <em>Fig. 6. Qualitative comparison on MSRS, showing pedestrian visibility and the preservation of background structures in a nighttime road scene.</em>
</p>

## 🎯 Downstream Object Detection

The downstream experiment uses **M3FD** to examine the utility of fused images for object detection with **YOLOv8**. Performance is evaluated using **Precision**, **Recall**, **mAP50**, and **mAP50:95**.

<p align="center">
  <img src="https://github.com/Wohaizainuli/BRIFusion/blob/main/Figure/Detect.jpg?raw=true" alt="YOLOv8 detection comparison on M3FD" width="100%">
</p>
<p align="center">
  <em>Fig. 7. Visual comparison of YOLOv8 object detection results using source images and fused images generated by different methods on M3FD.</em>
</p>

## 🗂️ Code Guide

| File | Description |
| :--- | :--- |
| `model/brifusion.py` | Main architecture, reliability heads, and decoders |
| `model/bri_layers.py` | DC-BRE, conditional experts, and RGCR |
| `dataloader/brifusion_data.py` | Paired data loading and image conversion |
| `utils/bri_losses.py` | Restoration, fusion, reliability, and consistency losses |
| `utils/degradations.py` | Synthetic degradation and local interventions |
| `train_main0.py` | Restoration and reliability pretraining entry point |
| `train_main1.py` | Joint fusion training entry point |
| `train_brifusion.py` | Shared training logic |
| `infer_brifusion.py` | Fusion inference and reliability visualization |
| `configs/brifusion.json` | Default model and loss configuration |

For the available ablation settings and their correspondence to the manuscript, see [Implementation Notes](docs/PAPER_ALIGNMENT.md).

## 🙏 Acknowledgments

This implementation builds on [DAMFusion](https://github.com/Wohaizainuli/DAMFusion). We thank the authors of LLVIP, M3FD, and MSRS for providing the datasets used in this study. See [LICENSE](LICENSE) for the repository license; dataset usage is governed by the respective dataset terms.

## 📬 Contact & Contributions

For questions about the method, implementation, or data preparation, please open an [issue](https://github.com/Wohaizainuli/BRIFusion/issues) or contact:

**Junjie Ma** — [junjiema_xmtra@163.com](mailto:junjiema_xmtra@163.com)

Bug reports, suggestions, and academic discussions are welcome. To contribute a code improvement, fork the repository and submit a pull request with a description of the changes.
