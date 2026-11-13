# Hiring Challenge Submission: Flowers-102 Classification
**Candidate Name**: Sanskriti Rai
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
- **Results**: Achieved **90.60%** accuracy on the test set, establishing a strong initial baseline.

---

## 3. Level 2: Intermediate Techniques
**Objective**: Improve performance with Data Augmentation and Regularization.

- **Approach**: Introduced a heavy augmentation pipeline to prevent overfitting and improve generalization.
- **Augmentation Pipeline**:
    - `RandomResizedCrop`: Forces the model to learn from parts of the flower.
    - `RandomRotation (15°)`: Invariance to orientation.
    - `ColorJitter`: Invariance to lighting conditions.
    - `RandomHorizontalFlip`.
- **Ablation Study**: Compared Baseline vs. Level 2 (Aug).
    - **Results**: Achieved **86.69%** accuracy.
    - **Analysis ("Regularization Tax")**: The accuracy is lower than the non-augmented baseline (90.60%). This is expected; heavy augmentation (random crops/jitter) creates a significantly harder training task. With a frozen backbone and a limited budget (15 epochs), the model has not yet fully converged on this harder distribution but learns more robust features that prevent overfitting in the long run and aid the ensemble.

---

## 4. Level 3: Advanced Architecture & Interpretability
**Objective**: Design a custom/advanced architecture and explain its decisions.

- **Architecture**: **EfficientNet-B0**.
    - **Reasoning**: EfficientNet scales depth, width, and resolution uniformly. It provides better parameter efficiency and accuracy on fine-grained tasks compared to ResNet.
    - **Training Strategy**: Fine-tuned the entire network (unfrozen) with a lower learning rate (`1e-4`) using the AdamW optimizer to refine features for flower textures.
- **Results**: Achieved **97.44%** accuracy, significantly outperforming ResNet50. This confirms the efficacy of modern architectures and fine-tuning.
- **Interpretability (Grad-CAM)**:
    - Implemented Gradient-weighted Class Activation Mapping (Grad-CAM).
    - **Visualization**: Generated heatmaps showing the model correctly focusing on the petals and distinctive floral features rather than the background.

---

## 5. Level 4: Expert Techniques (Ensemble)
**Objective**: Push accuracy using Ensemble Learning.

- **Approach**: **Soft Voting Ensemble**.
    - Combined predictions from **Model A (ResNet50 with augmentation, from Level 2: 86.69%)** and **Model B (EfficientNet-B0, from Level 3: 97.44%)**.
    - Averaged the Softmax probability distributions.
- **Reasoning**: ResNet and EfficientNet have different architectural biases (residual connections vs. MBConv blocks). Ensembling them captures complementary features, smoothing out individual model errors.
- **Results**: The ensemble achieved **97.68%** accuracy, surpassing the best individual model (EfficientNet's 97.44%). Detailed analysis shows the ensemble corrected predictions where the individual models were less confident.

---

## 6. Level 5: Production System
**Objective**: Optimization for Deployment (<100ms Inference).

- **Techniques**:
    1.  **Dynamic Quantization**: Converted the `nn.Linear` weights from FP32 to **INT8** with `torch.quantization.quantize_dynamic`.
        - *Result*: Model size went from 16.85 MB to 16.46 MB. The reduction is small because dynamic quantization only affects Linear layers, and EfficientNet-B0 is almost entirely convolutional. CPU latency still dropped from 188.23 ms to 55.68 ms.
    2.  **ONNX Export**: Converted the PyTorch model to Open Neural Network Exchange (ONNX) format with a dynamic batch axis, for cross-platform deployment.
- **Benchmarking Results** (single 224×224 image on CPU, averaged over 100 runs after 10 warm-up runs):

    | Variant | Model size | Latency |
    |---|---|---|
    | PyTorch FP32 | 16.85 MB | 188.23 ms |
    | PyTorch dynamic INT8 | 16.46 MB | 55.68 ms |
    | **ONNX Runtime** | – | **12.17 ms** |

    - **Latency**: ONNX Runtime met the **< 100ms** target with a wide margin, about **15× faster** than plain PyTorch.
- **Next step**: Static INT8 quantization or quantization-aware training would also compress the convolution layers and reduce model size.

---

## 7. Limitations & Failure Cases
- **Failure Cases**: The model sometimes confuses visually similar species (e.g., different types of Roses or Lilies) where the primary difference is only scale or minor texture.
- **Limitations**: The current resizing to 224x224 may lose fine-grained details present in the high-resolution original images. Increasing resolution (e.g., to 384x384) could improve accuracy at the cost of latency.

---

## 8. Code & Notebook Links
- **Repository**: [github.com/sanskreate/Terafac-Challenge](https://github.com/sanskreate/Terafac-Challenge)
- **Level 1 (Baseline)**: [level_1/Level_1_Baseline.ipynb](level_1/Level_1_Baseline.ipynb)
- **Level 2 (Augmentation)**: [level_2/Level_2_Augmentation.ipynb](level_2/Level_2_Augmentation.ipynb)
- **Level 3 (EfficientNet + Grad-CAM)**: [level_3/Level_3_Advanced_Arch.ipynb](level_3/Level_3_Advanced_Arch.ipynb)
- **Level 4 (Ensemble)**: [level_4/Level_4_Expert.ipynb](level_4/Level_4_Expert.ipynb)
- **Level 5 (Production)**: [level_5/Level_5_Production.ipynb](level_5/Level_5_Production.ipynb)
- **Shared data pipeline**: [shared/dataset.py](shared/dataset.py)