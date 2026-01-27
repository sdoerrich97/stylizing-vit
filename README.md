# Stylizing ViT: Anatomy-Preserving Instance Style Transfer for Domain Generalization @ ISBI 2026

<p align="center">
    [<a href="https://arxiv.org/abs/2601.17586v1">Preprint</a>]
    [<a href="">Publication</a>]
    [<a href="https://pypi.org/project/stylizing-vit/">PyPI</a>]
    [<a href="https://huggingface.co/collections/sdoerrich97/stylizing-vit">Pretrained Weights</a>]
    [<a href="#citation-">Citation</a>]
</p>


## Overview

Deep learning models in medical image analysis often struggle with generalizability across domains and demographic groups due to data heterogeneity and scarcity. Traditional augmentation improves robustness, but fails under substantial domain shifts. Recent advances in stylistic augmentation enhance domain generalization by varying image styles but fall short in terms of style diversity or by introducing artifacts into the generated images. To address these limitations, we propose *Stylizing ViT*, a novel Vision Transformer encoder that utilizes weight-shared attention blocks for both self- and cross-attention. This design allows the same attention block to maintain anatomical consistency through self-attention while performing style transfer via cross-attention. We assess the effectiveness of our method for domain generalization by employing it for data augmentation on three distinct image classification tasks in the context of histopathology and dermatology. Results demonstrate an improved robustness (up to 13% accuracy) over the state of the art while generating perceptually convincing images without artifacts. Additionally, we show that *Stylizing ViT* is effective beyond training, achieving a 17% performance improvement during inference when used for test-time augmentation.

![Overview of how Stylizing ViT is used to improve domain generalization during training and inference.](assets/method.png)
*Left: Overview of how Stylizing ViT is used to improve domain generalization during training and inference. During training, it generates stylistically diverse but anatomically consistent images, encouraging downstream classifiers to learn structure-aware representations. At test time, it is used to align unseen input styles with the training distribution, thereby mitigating style-induced domain shifts.*

*Right: Illustration of the underlying method. The input image pair ($I$, $S$) is processed by the encoder that fuses the anatomical structure of $I$ with the style characteristics of $S$ using cross-attention. Subsequently the stylized image $T$ is reconstructed through a two-layer MLP, a dot-product operation, and a convolutional layer. A frozen VGG19 encoder is used during training to compute perceptual losses.*

### Key Contributions

*   A modality-agnostic style augmentation method for domain generalization.
*   A novel ViT design enabling weight sharing within a unified attention block for both self- and cross-attention information fusion, termed *Stylizing ViT*.
*   Applicability for test-time augmentation (TTA) to enhance generalization to new, unseen domains during inference.
*   Comprehensive experiments across three medical imaging datasets, demonstrating consistent improvements in style transfer quality, state-of-the-art classification performance (up to +9% accuracy over prior best), and significant gains from TTA.

## Installation and Requirements

### From PyPI
```bash
pip install stylizing-vit
```

### From Source
Clone this repository and install it in editable mode:

```bash
git clone https://github.com/sdoerrich97/stylizing-vit.git
cd stylizing-vit
pip install -e .
```

To install training dependencies (like `wandb`, `accelerate`):
```bash
pip install -e ".[train]"
```

## Quick Start

### Inference (Style Transfer)

You can easily load a pre-trained model and perform style transfer on your images.

```python
import torch
from stylizing_vit import create_model, resize_image

# 1. Initialize Model & Load Pretrained Weights
# Weights are automatically downloaded from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model(backbone="base", weights="camelyon17wilds", train=False).to(device)
model.eval()

# 3. Stylize
# Assume content_img and style_img are normalized tensors (1, 3, 224, 224)
with torch.no_grad():
    stylized_img = model(content_img, style_img)
```

See [examples/inference_style_transfer.ipynb](examples/inference_style_transfer.ipynb) for a complete visual guide using the PathMNIST dataset.

### Training

The package exposes the core model and loss components, allowing you to integrate `StylizingViT` into your own training loops.

See [examples/training_demo.ipynb](examples/training_demo.ipynb) for a minimal training example.

## Model Zoo

We provide pretrained weights for the following configurations on our [Hugging Face Hub](https://huggingface.co/collections/sdoerrich97/stylizing-vit-6799059f0768407077a16087).

| Pathology | Dataset | Model Size | Weights Identifier |
| :--- | :--- | :--- | :--- |
| **Histopathology** | [Camelyon17-Wilds](https://wilds.stanford.edu/datasets/#camelyon17) | `base`, `small`, `tiny` | `camelyon17wilds` |
| | [Epithelium-Stroma](https://github.com/chenxinli001/Task-Aug) | `base`, `small`, `tiny` | `epistr` |
| **Dermatology** | [Fitzpatrick17k (Train: 12 / Val: 34 / Test: 56)](https://github.com/mattgroh/fitzpatrick17k) | `base`, `small`, `tiny` | `fitzpatrick17k_12_34_56` |
| | [Fitzpatrick17k (Train: 56 / Val: 34 / Test: 12)](https://github.com/mattgroh/fitzpatrick17k) | `base`, `small`, `tiny` | `fitzpatrick17k_65_43_21` |
| | [DDI (Train: 12 / Val: 34 / Test: 56)](https://ddi-dataset.github.io/) | `base`, `small`, `tiny` | `ddi_12_34_56` |
| | [DDI (Train: 56 / Val: 34 / Test: 12)](https://ddi-dataset.github.io/) | `base`, `small`, `tiny` | `ddi_65_43_21` |
| **Laparoscopy** | [Cholec80](https://zenodo.org/records/13170928) | `base`, `small`, `tiny` | `cholec80` |

You can load these directly using `create_model(backbone="<size>", weights="<Weights Identifier>")`. For example:
```python
model = create_model(backbone="base", weights="camelyon17wilds", train=False)
```

## Project Structure 📁

- `stylizing_vit/`: Core library package.
  - `model/`: Model architecture definition.
  - `loss/`: Loss functions.
  - `main.py`: High-level utilities and weight loading.
  - `util.py`: Image processing utilities.
- `examples/`: Jupyter notebooks demonstrating usage.
- `experiments/`: Experiments used within the paper.

## Results

![Qualitative results on training image pairs](assets/results.png)
*Qualitative results on training image pairs*

## Citation

If you use this code in your research, please cite: TBD

```bibtex
@article{doerrich2026stylizingvit,
  title={Stylizing ViT: Anatomy-Preserving Instance Style Transfer for Domain Generalization},
  author={Sebastian Doerrich and Francesco Di Salvo and Jonas Alle and Christian Ledig},
  year={2026},
  eprint={2601.17586},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

Apache-2.0
