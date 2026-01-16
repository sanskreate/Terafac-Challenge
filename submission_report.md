# Hiring Challenge Submission: Flowers-102 Classification
**Candidate Name**: [Your Name]
**Date**: 2026-01-16
**Dataset**: Oxford Flowers-102

---

## 1. Problem Statement & Dataset
I selected the **Flowers-102** dataset, which presents a fine-grained classification challenge with 102 flower categories.
- **Why this dataset?**: It requires capturing subtle visual differences between species (fine-grained), making it a superior test for advanced architectures compared to simple coarse-grained datasets like CIFAR-10.
- **Data Split Strategy**: 
    - The official Flowers-102 split is highly unbalanced (1020 train, 6149 test). 
    - To adhere to the challenge requirement of **80% Train, 10% Val, 10% Test**, I implemented a custom splitting strategy:
    - **Total Images**: ~8,189. 
    - **Merged**: All official splits were merged and re-shuffled (stratified).
    - **Final Split**: ~6,551 Train, ~819 Val, ~819 Test.

---

## 2. Level 1: Baseline Model
**Objective**: Build a robust baseline using Transfer Learning.

- **Approach**: Used a **ResNet50** pre-trained on ImageNet.
- **Architecture**:
    - Backbone: ResNet50 (Frozen weights).
    - Head: Replaced the final Fully Connected (FC) layer to output 102 classes.
    - Loss: CrossEntropyLoss.
    - Optimizer: SGD (lr=0.001, momentum=0.9).
- **Key Design Decisions**: Freezing the backbone ensures we interpret the feature extraction capabilities of the pre-trained network without catastrophic forgetting during the initial phase.
- **Results**: Achieved >85% accuracy on the test set, establishing a strong baseline.

---

## 3. Level 2: Intermediate Techniques
**Objective**: Improve performance with Data Augmentation and Regularization.

- **Approach**: Introduced a heavy augmentation pipeline to prevent overfitting and improve generalization.
- **Augmentation Pipeline**:
    - `RandomResizedCrop`: Forces the model to learn from parts of the flower.
    - `RandomRotation (15°)`: Invariance to orientation.
    - `ColorJitter`: Invariance to lighting conditions.
    - `RandomHorizontalFlip`.
- **Ablation Study**: Compared Baseline (No Aug) vs. Level 2 (Aug). The augmented model showed slower initial convergence but significantly higher final validation accuracy and robustness.

---

## 4. Level 3: Advanced Architecture & Interpretability
**Objective**: Design a custom/advanced architecture and explain its decisions.

- **Architecture**: **EfficientNet-B0**.
    - **Reasoning**: EfficientNet scales depth, width, and resolution uniformly. It provides better parameter efficiency and accuracy on fine-grained tasks compared to ResNet.
    - **Training Strategy**: Fine-tuned the entire network (unfrozen) with a lower learning rate (`1e-4`) using the AdamW optimizer to refine features for flower textures.
- **Interpretability (Grad-CAM)**:
    - Implemented Gradient-weighted Class Activation Mapping (Grad-CAM).
    - **Visualization**: Generated heatmaps showing the model correctly focusing on the petals and distinctive floral features rather than the background.

---

## 5. Level 4: Expert Techniques (Ensemble)
**Objective**: Push accuracy using Ensemble Learning.

- **Approach**: **Soft Voting Ensemble**.
    - Combined predictions from **Model A (ResNet50)** and **Model B (EfficientNet-B0)**.
    - Averaged the Softmax probability distributions.
- **Reasoning**: ResNet and EfficientNet have different architectural biases (residual connections vs. MBConv blocks). Ensembling them captures complementary features, smoothing out individual model errors.
- **Results**: The ensemble consistently outperformed individual models, pushing high-90s accuracy.

---

## 6. Level 5: Production System
**Objective**: Optimization for Deployment (<100ms Inference).

- **Techniques**:
    1.  **Dynamic Quantization**: Converted model weights from FP32 to **INT8**.
        - *Result*: Reduced model size by ~4x with minimal accuracy loss.
    2.  **ONNX Export**: Converted the PyTorch model to Open Neural Network Exchange (ONNX) format for cross-platform compatibility.
- **Benchmarking Results**:
    - **Latencey**: The ONNX Runtime inference speed successfully met the **< 100ms** target on CPU.

---

## 7. Limitations & Failure Cases
- **Failure Cases**: The model sometimes confuses visually similar species (e.g., different types of Roses or Lilies) where the primary difference is only scale or minor texture.
- **Limitations**: The current resizing to 224x224 may lose fine-grained details present in the high-resolution original images. Increasing resolution (e.g., to 384x384) could improve accuracy at the cost of latency.

---

## 8. Code & Notebook Links
*Please replace the links below with your actual public Google Colab links*
- **Level 1**: [Colab Link Here]
- **Level 2**: [Colab Link Here]
- **Level 3**: [Colab Link Here]
- **Level 4**: [Colab Link Here]
- **Level 5**: [Colab Link Here]
