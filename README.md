# Task-2


# Plant Disease Detection System 🌿

This repository implements a deep learning-based system to detect and classify plant leaf diseases. The project uses a Convolutional Neural Network (CNN) trained on this System, providing a user-friendly solution for farmers and agricultural experts to identify diseases and improve crop yield effectively.


## Overview
Plant diseases significantly affect global agricultural output, making early detection crucial. This system uses image classification techniques to accurately identify diseases in plant leaves.

### Key Objectives:
- Detect diseases such as *Early Blight*, *Powdery Mildew*, *Leaf Spot*, and classify healthy leaves.
- Provide actionable insights to prevent disease spread.

---

## Features
- **Deep Learning Model**: Uses Convolutional Neural Networks (CNN) for image classification.
- **Transfer Learning**: Includes pretrained models like MobileNet and ResNet for better accuracy and efficiency.
- **User-Friendly Interface**: Designed to integrate with mobile or web applications for ease of use.

---

## Requirements
Ensure you have the following dependencies installed:

### Libraries and Tools:
- Python 3.8+
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn

### Optional (for Transfer Learning):
- MobileNet
- ResNet

Install dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Usage

**Running Inference**:
Use the trained model to classify new leaf images:
```bash
python predict.py --image path/to/leaf_image.jpg
```

---

## Model Architecture
The default architecture includes:
- **Input Layer**: Accepts images resized to 224x224 pixels.
- **Convolutional Layers**: Feature extraction.
- **Pooling Layers**: Dimensionality reduction.
- **Fully Connected Layers**: Classification.
- **Activation**: ReLU and Softmax for output probabilities.

Optionally, pretrained models (MobileNet, ResNet) are available for transfer learning.

---

## Results
### Metrics:
- **Accuracy**: Achieved ~95% on test data.
- **Precision and Recall**: High scores for all classes.

### Visualization:
Confusion matrix and classification reports are available in the `results/` folder.
---

## Future Enhancements
- **Real-Time Detection**: Integrate with cameras for live analysis.
- **Mobile Deployment**: Export the model to TensorFlow Lite for Android/iOS apps.
- **Disease Severity Estimation**: Quantify the extent of leaf damage.

---

Feel free to reach out with feedback or suggestions. Let's work together to make agriculture smarter! 🌱
