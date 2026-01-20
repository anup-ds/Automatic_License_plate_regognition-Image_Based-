# Automatic Number Plate Recognition (ANPR) using Computer Vision & Deep Learning
 
 **Overview:**
Automatic Number Plate Recognition (ANPR) is a crucial component of modern intelligent transportation systems. This project presents an end-to-end ANPR pipeline that detects vehicle license plates from images, segments individual characters, and recognizes them using a custom-trained Convolutional Neural Network (CNN).

The system combines traditional Computer Vision techniques for efficient plate localization with Deep Learning for robust character recognition, achieving high accuracy on Indian license plates.

**Problem Statement:**

Manual vehicle monitoring for:
- Traffic law enforcement
- Toll collection
- Parking management
- Security access control is slow, error-prone, and not scalable.

Traffic cameras generate massive volumes of visual data, making manual review impractical. The challenge is to convert raw pixel data into structured, searchable license plate numbers automatically.

**Business Value:**

This ANPR system addresses real-world needs by enabling:
- ⚡ Faster Throughput – Automated toll and parking systems
- 🔐 Enhanced Security – Real-time identification of authorized / unauthorized vehicles
- 💰 Cost Reduction – Reduced manpower for 24×7 monitoring
- 📈 Scalability – Deployable on existing CCTV infrastructure

**System Architecture:**

The solution follows a modular pipeline:
1. License Plate Detection
   - Haar Cascade Classifier for fast and lightweight localization
2. Character Segmentation
   - mage preprocessing (Grayscale, Thresholding)
   - Contour detection and geometric filtering
3. Character Recognition
   - Custom CNN trained on alphanumeric characters (0–9, A–Z)
4. Inference Engine
   - Converts segmented characters into final license plate text

**Methodology**
1️. Image Preprocessing
   - Grayscale conversion to reduce dimensionality
   - Image resizing for standardization
   - Binary inverse thresholding with Otsu’s method for high contrast

2️. License Plate Detection
   - Pre-trained Haar Cascade (haarcascade_russian_plate_number.xml)
   - Multi-scale detection for varying plate sizes
   - Cropping detected plate region for downstream processing

3️. Character Segmentation
   - Contour detection using OpenCV
   - Filtering based on:
     - Aspect ratio
     - Area
   - Sorting characters left-to-right to preserve sequence

4️. CNN Model for Character Recognition
   - Architecture Highlights:
     - Input: 28×28 grayscale images
     - 3 Convolutional layers (32 → 64 → 128 filters)
     - MaxPooling after each convolution
     - Fully connected Dense layer (128 neurons)
     - Dropout (0.5) to prevent overfitting
     - Output: 36 classes (digits + alphabets)

  - Training Details:
    - Optimizer: Adam
    - Loss Function: Sparse Categorical Crossentropy
    - Epochs: 30
    - Model saved as: char_recognition_cnn.h5

**Results & Performance:**

 - Validation Accuracy: ~99%
 - Training Accuracy: ~95%
 - Minimal overfitting observed
 - High precision and recall across most character classes

**Known Limitations:**

 - Visual ambiguity between:
 - 0 and O
 - 1 and I
   (A common OCR challenge)

**Testing & Visual Validation**
The system was tested on unseen vehicle images, successfully:
 - Detecting license plates
 - Segmenting characters
 - Predicting complete license numbers accurately

Bounding boxes and predicted text are displayed for qualitative validation.

📁 Dataset & Resources

 - Dataset: AI Indian License Plate Recognition Data (Kaggle)-https://www.kaggle.com/code/sarthakvajpayee/license-plate-recognition-using-cnn/notebook
 - Contains segmented character images (0–9, A–Z)
 - Includes pre-trained Haar Cascade XML file

**Tech Stack**

 - Python
 - OpenCV – Image processing & detection
 - TensorFlow / Keras – CNN modeling
 - Scikit-learn – Evaluation metrics
 - Matplotlib / Seaborn – Visualization
 - Google Colab – GPU-accelerated training

**Future Improvements**

 - Replace Haar Cascades with YOLO for better detection in complex scenes
 - Add Optical Character Verification (OCV) using plate format rules
 - Optimize with TensorFlow Lite for edge deployment (Raspberry Pi)
