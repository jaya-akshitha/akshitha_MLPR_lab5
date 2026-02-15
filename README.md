# Lab 5 – Face Detection and Clustering using HSV Features

## 📌 Project Overview

This project implements face detection, feature extraction, clustering, and template matching using classical computer vision and machine learning techniques.

The system detects faces in an image, extracts color-based features (Hue and Saturation), groups similar faces using K-Means clustering, and identifies the closest match to a given template image using distance metrics.

The objective of this lab is to understand how image processing and unsupervised learning techniques can be combined for practical pattern recognition tasks.

---

## 🎯 Objectives

- Detect faces using Haar Cascade Classifier.
- Extract meaningful color features from detected faces.
- Perform unsupervised clustering using K-Means.
- Visualize clusters for analysis.
- Match a template face using similarity metrics.
- Understand the role of distance measures in classification.

---

## 🛠️ Tools & Libraries Used

- OpenCV (cv2) – Face detection and image processing
- NumPy – Numerical operations
- Matplotlib – Data visualization
- Scikit-learn – K-Means clustering
- SciPy – Distance computation

---

## 📂 Methodology

### 1️⃣ Image Preprocessing
- Read input image.
- Convert from BGR to grayscale for face detection.
- Convert to HSV color space for feature extraction.

### 2️⃣ Face Detection
- Use Haar Cascade classifier.
- Detect face regions.
- Draw bounding boxes around detected faces.

### 3️⃣ Feature Extraction
For each detected face:
- Compute mean Hue value.
- Compute mean Saturation value.
- Store features as a 2D feature vector.

These features represent the dominant color characteristics of each face.

### 4️⃣ K-Means Clustering
- Apply K-Means algorithm (k=2).
- Group faces based on similarity in HSV space.
- Visualize clusters using a scatter plot.

### 5️⃣ Template Matching
- Detect face in template image.
- Extract HSV features.
- Compute distance between template and detected faces.
- Identify closest matching face using minimum distance.

---

## 📊 Distance Metrics Explained

The following distance metrics are commonly used in classification and similarity matching:

- **Euclidean Distance** – Straight-line distance between two points.
- **Manhattan Distance** – Sum of absolute differences.
- **Minkowski Distance** – Generalized distance metric.
- **Mahalanobis Distance** – Accounts for variance and correlation.
- **Cosine Distance** – Measures angle between vectors.
- **Hamming Distance** – Used for categorical/binary data.

In this lab, Euclidean distance is primarily used for similarity comparison.

---

## 📈 Results

- Faces were successfully detected using Haar Cascade.
- HSV features effectively represented facial color characteristics.
- K-Means clustering grouped similar faces together.
- Template matching correctly identified the closest face based on feature similarity.
- Scatter plot visualization provided clear cluster separation.

---

## ✅ Conclusion

This lab demonstrates how classical computer vision techniques combined with basic machine learning algorithms can solve real-world recognition problems.

Key learnings:

- Haar Cascade provides efficient real-time face detection.
- HSV color space is useful for extracting meaningful color features.
- K-Means clustering helps in grouping similar feature vectors without supervision.
- Distance metrics play a crucial role in similarity-based classification.
- Even simple features like mean Hue and Saturation can be effective for clustering tasks.

Overall, the project highlights the integration of image processing, feature engineering, unsupervised learning, and similarity matching in a complete pipeline.

---

## 🚀 How to Run the Project

1. Install required libraries:

   pip install opencv-python numpy matplotlib scikit-learn scipy

2. Update image file paths in the notebook.

3. Run all cells sequentially.

4. Observe:
   - Face detection output
   - Cluster visualization
   - Template matching result

---

## 📌 Limitations

- Haar Cascade may fail under extreme lighting or occlusions.
- Only color-based features are used (no texture or shape features).
- K-Means requires predefined number of clusters.
- Performance may vary with different datasets.

---

## 🔮 Future Improvements

- Use deep learning-based face detectors (e.g., CNN models).
- Extract additional features (LBP, HOG, embeddings).
- Implement automatic cluster selection.
- Improve robustness under varying lighting conditions.
- Extend to multi-class face recognition.

---

## 👩‍💻 Author

Akshitha  
Computer Vision / Image Processing Lab  
