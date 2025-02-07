# PlantDoc: AI-driven-Support-Tool

## Overview

This project implements a deep learning-based approach for classifying plant diseases using a pretrained MobileNet model. The dataset consists of images of plant leaves, categorized into 38 disease classes, including healthy samples. The model is trained using PyTorch and fine-tuned with transfer learning, achieving a high accuracy of **99.61%**, which is a **0.41% improvement** over an existing model performing the same task. This improvement was made possible by enhancing the model and utilizing the dataset from Atharva Ingle's Kaggle notebook.

---

## Dataset

The dataset used for training and validation contains:

- **Total images**: 70,295
- **Total disease classes**: 38
- **Unique plant species**: 14
- **Healthy class count**: 12
- **Total test images**: 8,786

Each image is preprocessed and resized to **256x256 pixels**, converted into tensors, and normalized before feeding into the model.

---

## Model Selection

To identify the best-performing model for this task, we evaluated multiple pretrained architectures:

- **ResNet-18**
- **VGG16**
- **DenseNet121**
- **MobileNetV2**

### Model Performance Comparison

| Model       | Validation Loss | Validation Accuracy |
|------------|----------------|----------------------|
| MobileNet  | 3.69           | **99.61%**          |
| DenseNet   | 3.75           | 98.02%              |
| ResNet     | 3.84           | 97.56%              |
| VGG16      | 3.67           | 97.20%              |

MobileNet outperformed the other models in terms of accuracy and efficiency, making it the final choice for deployment.

---

## Data Preprocessing

### Loading & Cleaning:
- Dataset loaded using `torchvision.datasets.ImageFolder`.
- Duplicate or misclassified data identified and removed.

### Normalization & Augmentation:
- Images converted to tensors and pixel values scaled to **[0,1]**.
- Data augmentation applied (**random flips, rotations, etc.**) to improve generalization.

### Splitting Dataset:
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

---

## Training & Optimization

- **Optimizer**: Adam
- **Batch Size**: 32
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate Scheduler**: One Cycle Policy
- **Gradient Clipping**: Applied at 0.1 to prevent gradient explosion

Training consisted of **4 epochs**, utilizing **One Cycle Learning Rate Scheduling** for optimized convergence.

### Training Metrics:

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
|-------|-----------|----------------|----------------------|
| 1     | 0.2455    | 0.5532         | 83.29%               |
| 2     | 0.2680    | 0.3915         | 87.40%               |
| 3     | 0.1303    | 0.0514         | 98.37%               |
| 4     | 0.0360    | 0.0166         | **99.52%**           |

---

## Model Evaluation

- **Confusion Matrix Analysis**: Demonstrated near-perfect classification across all classes, with minimal misclassification occurring between visually similar diseases.
- **Precision, Recall, F1-Score**: **99%+ for all classes**.
- **Prediction Accuracy on Test Set**: **99.61%**

---

## Deployment

The final model is saved as `mobilenet_weights.pth` for use in inference applications.

### Running Inference:

```python
import torch
from torchvision import transforms
from PIL import Image
from model import PretrainedModel

# Load model
model = PretrainedModel('mobilenet', num_classes=38)
model.load_state_dict(torch.load("mobilenet_weights.pth"))
model.eval()

# Preprocess input image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path)
    return transform(image).unsqueeze(0)

# Predict function
def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, dim=1)
    return predicted_class.item()
```

---

## Key Takeaways

- **MobileNetV2** proved to be the most effective model in balancing accuracy and efficiency.
- **One Cycle Learning Rate Scheduling** accelerated convergence and improved performance.
- **Careful preprocessing and data augmentation** significantly contributed to achieving high accuracy.
- **The model is lightweight** and optimized for deployment on edge devices.

---

## Future Work

- Implementing a **real-time inference API** for mobile applications.
- Expanding the dataset to **include more plant species**.
- Exploring **self-supervised learning** for improved generalization.
