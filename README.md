# 🌸 Flowers-102 Fine-Grained Image Classification: From Baseline to Production

A five-level computer-vision project, built for the Terafac hiring challenge. It starts with a transfer-learning baseline, then adds augmentation, a stronger architecture with explainability, an ensemble, and finally a model optimized for fast CPU inference.

| | |
|---|---|
| **Task** | Fine-grained classification of **102 flower species** |
| **Dataset** | [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) (8,189 images) |
| **Best accuracy** | **97.68%** on the held-out test set (ResNet50 + EfficientNet-B0 ensemble) |
| **Fastest inference** | **12.17 ms per image on CPU** with ONNX Runtime (vs 188.23 ms in plain PyTorch) |
| **Stack** | PyTorch · torchvision · scikit-learn · OpenCV · ONNX · ONNX Runtime |

---

## 📊 Results at a glance

| Level | Model | What changed | Test accuracy |
|---|---|---|---|
| 1 · Baseline | ResNet50 (frozen backbone) | Transfer learning, new 102-class head | **90.60%** ¹ |
| 2 · Augmentation | ResNet50 (frozen backbone) | Heavy augmentation + ablation vs Level 1 | **86.69%** |
| 3 · Advanced architecture | EfficientNet-B0 (fully fine-tuned) | Better architecture + Grad-CAM explainability | **97.44%** |
| 4 · Ensemble | ResNet50 (L2) + EfficientNet-B0 (L3) | Soft-voting ensemble | **97.68%** 🏆 |
| 5 · Production | EfficientNet-B0 → INT8 / ONNX | Quantization, ONNX export, latency benchmark | **12.17 ms / image** |

¹ The Level 2 notebook re-evaluates the saved Level 1 model on the test set and gets 90.60%. The final cell of the Level 1 training run printed 90.11%.

---

## 🗂️ Data split

The official Flowers-102 split is unusual: 1,020 training images and 6,149 test images. The challenge required an **80 / 10 / 10** split, so all official splits are **merged and re-split with stratification** (`random_state=42`), which keeps every class in the same proportion in each split:

| Split | Images |
|---|---|
| Train | 6,551 |
| Validation | 819 |
| Test | 819 |

All images are resized to 256 × 256, center-cropped to 224 × 224 and normalized with ImageNet statistics. See [`shared/dataset.py`](shared/dataset.py).

---

## 🔬 The five levels

### Level 1: Baseline (`level_1/Level_1_Baseline.ipynb`)
- **Model:** ResNet50 pre-trained on ImageNet. The backbone is frozen, and only a new fully connected layer with 102 outputs is trained.
- **Training:** CrossEntropyLoss, SGD (lr = 0.001, momentum = 0.9), 10 epochs, no augmentation.
- **Why:** freezing the backbone shows how much the pre-trained ImageNet features alone can do, and gives a fast, stable reference point.
- **Result:** 90.60% test accuracy. Validation accuracy climbs steadily from 42% after epoch 1 to about 89%.

### Level 2: Data augmentation and ablation (`level_2/Level_2_Augmentation.ipynb`)
- **Augmentation pipeline:** `RandomResizedCrop(224, scale=0.8–1.0)`, `RandomHorizontalFlip`, `RandomRotation(15°)`, `ColorJitter(0.2, 0.2, 0.2, 0.1)`.
- **Training:** same frozen ResNet50 and optimizer, 15 epochs.
- **Ablation:** compared directly against the Level 1 model on the same test set.
- **Result:** 86.69%, which is **3.91 points below** the baseline.
- **Analysis ("regularization tax"):** with a frozen backbone and a short training budget, heavy augmentation makes the task harder than the model can absorb. The features it learns are more robust, though, and that diversity pays off in the Level 4 ensemble.

### Level 3: Advanced architecture and explainability (`level_3/Level_3_Advanced_Arch.ipynb`)
- **Model:** EfficientNet-B0, **fully fine-tuned** (all layers unfrozen), with a new 102-class classifier.
- **Training:** AdamW (lr = 1e-4, weight decay = 1e-4), 15 epochs, with augmentation. The best checkpoint by validation accuracy is saved.
- **Why EfficientNet:** it scales depth, width and resolution together, so it gets more accuracy per parameter. That matters when the differences between flower species are subtle textures.
- **Result:** **97.44%** test accuracy, **+6.8 points** over the baseline. Validation accuracy passed 98%.
- **Explainability, Grad-CAM:** a custom `GradCAM` class hooks the last convolutional block and produces heatmaps. They show the model looking at **petals and floral structure, not the background**.

### Level 4: Ensemble (`level_4/Level_4_Expert.ipynb`)
- **Method:** a soft-voting ensemble that averages the softmax probabilities of the **Level 2 ResNet50** and the **Level 3 EfficientNet-B0**.
- **Why it works:** residual networks and MBConv-based EfficientNets have different inductive biases, so their mistakes don't fully overlap.

| Model | Test accuracy |
|---|---|
| ResNet50 (Level 2) | 86.69% |
| EfficientNet-B0 (Level 3) | 97.44% |
| **Soft-voting ensemble** | **97.68%** |

### Level 5: Production optimization (`level_5/Level_5_Production.ipynb`)
Target: **under 100 ms inference on CPU**.

1. **Dynamic quantization** (`torch.quantization.quantize_dynamic`): `nn.Linear` weights converted from FP32 to INT8.
2. **ONNX export**: the model is exported to `models/level_5_model.onnx` with a dynamic batch axis, for cross-platform serving.
3. **Benchmark**: 10 warm-up runs, then the average over 100 runs on a single 224 × 224 image on CPU.

| Variant | Size | CPU latency |
|---|---|---|
| PyTorch FP32 | 16.85 MB | 188.23 ms |
| PyTorch dynamic INT8 | 16.46 MB | 55.68 ms |
| **ONNX Runtime** | – | **12.17 ms** ✅ |

The ONNX Runtime version is about **15× faster** than plain PyTorch and well under the 100 ms target. Model size barely changes because dynamic quantization only touches the `Linear` layers, and almost all of EfficientNet-B0 is convolutions. Static quantization or quantization-aware training would be needed to shrink the convolution weights too.

---

## 🧱 Project structure

```
Terafac-Challenge/
├── level_1/Level_1_Baseline.ipynb        # ResNet50 transfer-learning baseline
├── level_2/Level_2_Augmentation.ipynb    # Augmentation + ablation vs baseline
├── level_3/Level_3_Advanced_Arch.ipynb   # EfficientNet-B0 fine-tuning + Grad-CAM
├── level_4/Level_4_Expert.ipynb          # Soft-voting ensemble
├── level_5/Level_5_Production.ipynb      # INT8 quantization, ONNX export, latency benchmark
├── shared/
│   ├── dataset.py                        # Merged + stratified 80/10/10 Flowers-102 loaders
│   └── visualization.py                  # Training curves and image display helpers
├── submission_report.md                  # Written challenge report
└── requirements.txt
```

Datasets, checkpoints (`*.pth`) and ONNX files are git-ignored. They're created when you run the notebooks.

---

## 🚀 Getting started

```bash
git clone https://github.com/sanskreate/Terafac-Challenge.git
cd Terafac-Challenge
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks **in order, from inside each level's folder**, because later levels load earlier checkpoints using relative paths:

1. `level_1`: downloads Flowers-102 into `./data` through torchvision, then trains the baseline.
2. `level_2`: trains the augmented ResNet50 → `level_2/models/level_2_augmented.pth`.
3. `level_3`: fine-tunes EfficientNet-B0 → `level_3/models/level_3_efficientnet.pth`.
4. `level_4`: loads both checkpoints and evaluates the ensemble.
5. `level_5`: loads the Level 3 model, then quantizes, exports and benchmarks it.

A GPU speeds up training a lot, but everything also runs on CPU. The results above were produced on CPU.

---

## ⚠️ Limitations and next steps

- **Confusable species:** the model sometimes mixes up visually similar species, such as different roses or lilies, where the difference is mostly scale or fine texture.
- **Resolution:** downscaling to 224 × 224 throws away fine detail. Training at 384 × 384 should improve accuracy, at some cost in latency.
- **Augmentation budget:** Level 2 would likely beat the baseline with more epochs or an unfrozen backbone.
- **Compression:** static INT8 quantization or pruning would shrink the convolution layers, which dynamic quantization leaves untouched.

---

👩‍💻 **Author:** [Sanskriti Rai](https://www.linkedin.com/in/sanskreate/) · [sanskritirai.live](https://sanskritirai.live)
