# Lab 5 – Face Detection and Clustering using HSV Features

## 📌 Overview
This lab focuses on detecting faces in an image, extracting color-based features, clustering the detected faces using K-Means, and performing template matching using distance metrics.

The implementation uses OpenCV for image processing, NumPy for numerical operations, Matplotlib for visualization, and Scikit-learn for clustering.

---

## 🧠 Objectives

- Detect faces in an image using Haar Cascade classifier.
- Extract Hue and Saturation features from detected faces.
- Perform clustering using K-Means algorithm.
- Visualize clusters using scatter plots.
- Match a template face image with detected faces using distance metrics.
- Understand commonly used distance metrics in classification.

---

## 🛠️ Technologies & Libraries Used

- OpenCV (`cv2`) – Image processing and face detection  
- NumPy – Numerical computations  
- Matplotlib – Data visualization  
- Scikit-learn (KMeans) – Clustering algorithm  
- SciPy – Distance calculations  

---

## 📂 Project Workflow

### 1️⃣ Image Reading
- Read the main image containing multiple faces.
- Convert the image from BGR to Grayscale for face detection.

### 2️⃣ Face Detection
- Load Haar Cascade classifier.
- Detect faces in the image.
- Store face coordinates.

### 3️⃣ Feature Extraction
- Convert image to HSV color space.
- Extract:
  - Mean Hue
  - Mean Saturation
- Store features for each detected face.

### 4️⃣ Clustering
- Apply K-Means clustering on Hue and Saturation features.
- Divide faces into two clusters.
- Plot clusters using a scatter plot.

### 5️⃣ Template Matching
- Read a template face image.
- Detect face in template.
- Extract Hue and Saturation features.
- Compare template features with clustered faces using distance metrics.
- Identify closest matching face.

---

## 📊 Distance Metrics Used

Common distance metrics in classification include:

- Euclidean Distance  
- Manhattan Distance  
- Minkowski Distance  
- Mahalanobis Distance  
- Cosine Distance  
- Hamming Distance  

---

## 📈 Output

- Detected faces highlighted.
- Scatter plot showing clustered faces.
- Template face comparison result.
- Identification of closest matching face based on feature distance.

---

## 🚀 How to Run

1. Install required libraries:
   ```bash
   pip install opencv-python numpy matplotlib scikit-learn scipy
