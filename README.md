# GeoStateNet 🗺️

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://img.shields.io/badge/Colab-Open%20Notebook-orange?logo=google-colab)](https://colab.research.google.com/github/dcrew44/GeoStateNet/blob/main/colab_training_example.ipynb)



A deep learning model that achieves **76.66% accuracy** in predicting US states from street view images. This project demonstrates advanced computer vision techniques applied to geographical feature recognition. 


## 🎯 Project Overview

GeoStateNet is a fine-tuned ResNet-101 model with a custom head, trained on the 50States10K dataset to classify which US state a street view image was taken in. Developed as my senior project at the University of Redlands, this project explores how deep learning can recognize subtle geographical patterns in visual data.

### Key Features
- **76.66% panorama accuracy** - Significantly outperforming the original DeepGeo baseline (38.32%)
- **Real-time inference** - Integrated with GeoGuessr game via Chrome extension
- **Comprehensive training pipeline** - Multi-phase training with Weights & Biases integration
- **Practical Implementation** - Includes API server and browser extension for practical applications

## 🏆 Results

| Model | Panorama Accuracy | Single Image Accuracy | 
|-------|-------------------|----------------------|
| DeepGeo (baseline) | 38.32% | 25.92% |
| **GeoStateNet** | **76.66%** | **60.82%** |

### Training Metrics (via Weights & Biases)
- Final validation loss: 1.34
- Best epoch: 4/5
- Training conducted on Google Colab with NVIDIA A100 GPU

**Details on training runs available upon request**

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/dcrew44/GeoStateNet.git
cd GeoStateNet

# Install the package
pip install -e .
```

### Download Pre-trained Model

```bash
# Download the trained model checkpoint (76% accuracy)
wget https://github.com/dcrew44/GeoStateNet/releases/download/v1.0/best_model.pth -O checkpoints/best_model.pth
```

### Running Inference

```bash
# Quick inference on a single image
python predict.py street_view.jpg

# Use a specific checkpoint
python predict.py street_view.jpg --checkpoint path/to/model.pth

# Show top 10 predictions
python predict.py street_view.jpg --top-k 10
```

## 🎮 GeoGuessr Integration

This project includes a complete ecosystem for real-time state prediction in GeoGuessr:

### Related Repositories
- **[GeoStateNet-API](https://github.com/dcrew44/geostatenet-api)** - FastAPI server for model inference
- **[GeoStateNet-Extension](https://github.com/dcrew44/geostatenet-extension)** - Chrome extension for GeoGuessr integration


## 📊 Dataset

This project uses the 50States10K and 50States2K datasets from the DeepGeo paper:

- **Training**: 500K images (2.5K locations × 4 views × 50 states)
- **Testing**: 100K images (500 locations × 4 views × 50 states)
- **Resolution**: 256×256 pixels
- **Coverage**: All 50 US states with stratified sampling

### Obtaining the Dataset

1. Download from the DeepGeo project:
   - [50States10K (Training)](https://drive.google.com/file/d/1Y8eqx1Uy8kuRP4BNmTCVvrNbCxx6RoiP/view)
   - [50States2K (Test)](https://drive.google.com/file/d/1zHc3wcQxicoij0tGHBpVSKqN_QgZsF-t/view)

2. Extract to the `data/` directory following the structure in the repository


**Licence note:** The 50States10K/2K datasets are released by the DeepGeo authors without an explicit licence. Redistribution here is limited to download links; users should cite the original paper when using the data.


## 🏗️ Architecture

GeoStateNet employs a sophisticated training strategy:

1. **Base Model**: ResNet-101 pretrained on ImageNet
2. **Custom Head**: AdaptiveConcatPool2d + Fully Connected layers with dropout
3. **Multi-Phase Training**:
   - Phase 1: Head-only training 
   - Phase 2: Unfreeze Layer 4 
   - Phase 3: Unfreeze Layers 2-4 
4. **Panorama Aggregation during Inference**: Averages predictions across 4 cardinal directions

## 📚 Credits & References

This project takes inspiration and implements techniques from several key works:

- Suresh, Chodosh, Abello (2018). **[DeepGeo: Photo Localization with Deep Neural Network.](https://arxiv.org/abs/1810.03077)** 
  - Provided foundational dataset and baseline results

- Victor De Fontnouvelle. **[GeoGuessrBot: Predicting the Location of Any Street View Image.](https://vdefont.github.io/2021/06/20/geoguessr.html?utm_source=catalyzex.com)**
  - Demonstrated effectiveness of averaging predictions across panorama views.
  - This technique was crucial for achieving 76% accuracy on test dataset.

- Haas, Skreta, Alberti, Finn (2024). **[Pigeon: Predicting Image Geolocations](https://arxiv.org/abs/2307.05845)**
  - Current SOTA in image geolocation.
  - Inspired chrome extension for playing GeoGuessr

- Jeremy Howard and Sylvain Gugger. **[FastAI](https://github.com/fastai/fastai)** 
  - I implemented training best practices from the fastai library including:
    - AdaptiveConcatPool2d implementation
    - Discriminative learning rates
    - One-cycle learning rate scheduling
    - Progressive unfreezing strategy
    - Extensive augmentation pipline

### Technical Implementation
- **PyTorch** and **torchvision** for model implementation
- **Weights & Biases** for experiment tracking
- **Google Colab** for GPU training resources

## 🔬 Technical Details

### Training Configuration
```yaml
# Key hyperparameters that achieved 76% accuracy
optimizer: AdamW
batch_size: 256
learning_rates:
  - Phase 1: 0.01 (head only)
  - Phase 3: 0.004 (unfreeze layer2-4)
weight_decay: 0.01
label_smoothing: 0.1
```

## 🛠️ Development

### Running Training

```bash
# Local training (requires GPU)
python -m state_classifier.main --config config.yaml

# For Google Colab, see colab_training_example.ipynb
```

### Experiment Tracking

All experiments are tracked via Weights & Biases. Key metrics include:
- Per-state accuracy breakdown
- Confusion matrices
- Training/validation curves


### Future Work
- 🌐 Multi-view transformer that learns cross-directional context  
- 🗺️ Fine-grained geocells (≈12 km) using contrastive pre-training  
- 📊 Per-state calibration plots & model card for bias analysis



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

Special thanks to:
- My advisor at the University of Redlands Professor Rick Cornez 
- The authors of all the works which inspired me
- A special thanks to the authors of the DeepGeo paper for making their dataset publicly available
- The open-source community for invaluable tools and inspiration
- Professor Joanna Bieri who inspired me to pursue machine learning

---


<p align="center">
  <b>If you found this project interesting, please consider giving it a ⭐!</b>
</p>

<p align="center">
  <b>Made&nbsp;with&nbsp;☕ by&nbsp;Hayden&nbsp;Barolette</b><br>
  <a href="https://www.linkedin.com/in/hayden-barolette-702476156">
    <img src="https://img.shields.io/badge/LinkedIn-connect-blue?logo=linkedin" alt="LinkedIn Badge">
  </a>
</p>
