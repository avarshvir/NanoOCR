# NanoOCR - Optical Character Recognition

OCR model to detect text.

---

### Dataset
- Source: Kaggle [Link](https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset)
- Size: ~45,500 character-level images
- Classes: 36 total (Letters A-Z, Numbers 0-9)
- Format: Grayscale, pre-processed to 32x32 pixels.

---

### Architecture
- Total Parameters: 626,916 (Total model size is only ~2.4 MB on disk)
- Block 1: Conv2d (32 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
- Block 2: Conv2d (64 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
- Block 3: Conv2d (128 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)
- Classifier Head: Flatten -> Linear Dense (256 units) -> ReLU -> Dropout (0.3) -> Linear Output (36 units)

---

### Training Technique
- Optimizer: AdamW (Learning rate: 0.001, Weight decay: 1e-4 for regularization)
- Loss Function: Multi-class Cross-Entropy Loss
- Hardware: Trained on a single T4 GPU (Google Colab)
- Techniques used: Batch Normalization to speed up convergence, and a 30% Dropout rate to prevent overfitting on specific fonts.

---

### Metrics and Performance
- Training Time: ~45 seconds total (only 5 epochs required)
- Training Accuracy: 97.22%
- Validation Accuracy: 97.46% (It generalized perfectly to unseen data!)
- Inference Latency: Near-zero milliseconds per character using ONNX CPU Execution Provider.

---

### Deployment Pipeline
- Image Processing: OpenCV (Gaussian Blur + Adaptive Thresholding for binarization).
- Segmentation: OpenCV findContours with left-to-right bounding box sorting.
- Backend: FastAPI with asynccontextmanager for memory-efficient model loading.
- Frontend: Vanilla JavaScript + HTML + Tailwind CSS.

---

Feel free to star the repo and improve the project as well as model :)