# **Medicinal Plant Classification Using Machine Learning** 

## **Medicinal Plant Classification Using SVM**

A machine learning–based application for classifying medicinal plant leaves using feature engineering + PCA + SVM, deployed through a Streamlit interface.

## **📌Project Overview** :page_facing_up:
Medicinal plants play a vital role in traditional medicine systems such as Ayurveda, Siddha, and Homeopathy. Accurate identification of these plants is essential to ensure the safety and effectiveness of herbal formulations.

This project presents an automated Medicinal Plant Classification System capable of identifying leaf species using classical image processing and machine learning techniques. The system extracts texture, shape, color, and gradient-based features from uploaded leaf images and classifies them using a trained Support Vector Machine (SVM) model.

The project includes:

✔ Leaf image preprocessing
✔ HSV-based segmentation
✔ Texture feature extraction (LBP, GLCM, Gabor)
✔ Gradient-based structural features
✔ Color moment analysis
✔ PCA for dimensionality reduction
✔ SVM-based classification
✔ Streamlit GUI for real-time predictions
✔ Built-in medicinal plant information system

---

## **Data Collection** :package:
The dataset comprises images of 30 different plant species, each labeled with both common and botanical names. These images vary in size, providing a comprehensive dataset for robust classification tasks.

- **Dataset URL:** [Medicinal Leaf Dataset](https://data.mendeley.com/datasets/nnytj2v3n5/1)

---

## **Preprocessing** :hammer_and_wrench:
### **1. Segmentation** :scissors:
Images are segmented using the HSV (Hue, Saturation, Value) color space to isolate the relevant plant regions, removing unnecessary background.

<div>
  <img src="https://github.com/user-attachments/assets/aee5227f-2b46-44f1-a2ed-b34d91959011" alt="ImgSeg" width="400" />
</div>

### **2. Gray Scale Conversion** :black_circle:
The segmented images are then converted to grayscale to simplify further processing while preserving essential details.

<div>
  <img src="https://github.com/user-attachments/assets/38c27710-bd6f-42bb-a8ba-8b07ea5af34b" alt="GrayScale" width="400" />
</div>

### **3. Sobel Filter** :triangular_ruler:
A Sobel Filter is applied to highlight the edges in the images, making them more suitable for feature extraction.

<div>
  <img src="https://github.com/user-attachments/assets/a807095a-0e98-466c-9356-e4488c332393" alt="Sobel" width="400" />
</div>

---

## **Feature Extraction** :mag_right:
Features are extracted from both segmented and grayscale images using the following techniques:

### **1. Local Binary Patterns (LBP)** :1234:
LBP converts an image into a binary pattern by comparing neighboring pixels to the center pixel. This binary code is then converted to a decimal value, serving as a texture descriptor.

### **2. Gray-Level Co-occurrence Matrix (GLCM)** :chart_with_upwards_trend:
GLCM analyzes pixel pairs within a specific spatial relationship, creating a matrix that reflects their frequency. This matrix is used to derive texture features such as contrast, correlation, energy, and homogeneity.

### **3. Gabor Filters** :wave:
Gabor filters process the image by convolving it with a sinusoidal wave modulated by a Gaussian envelope. These filters are sensitive to specific image features, such as edges and textures, at various scales and orientations.

### **4. Color Moments** :rainbow:
Color moments, including mean, standard deviation, and skewness, capture the color distribution in each channel (e.g., RGB). These moments are highly effective for image classification tasks.



---

## **Dimensionality Reduction** :scissors:
To improve model performance and reduce dimensionality, the dataset is split into training and test sets, and the following technique is applied:

### **Principal Component Analysis (PCA)** :bar_chart:
PCA identifies the principal components that capture the most variance in the data, reducing the dimensionality while retaining essential information.

---

## **Model Performance Summary** :computer:
During experiments:

SVM (Linear Kernel)	84.42%,
KNN	80.50%,
Random Forest	76.62%

The linear SVM clearly outperformed other classifiers for this dataset.

---

## **Prediction** :crystal_ball:
Real-time data can be used for prediction by leveraging pre-trained models, including PCA for dimensionality reduction and SVM for classification. The process involves:

1. **Data Preprocessing:** Incoming data is normalized using the saved StandardScaler to ensure consistency.
2. **Dimensionality Reduction:** The data is transformed using the saved PCA model, retaining the most significant features.
3. **Prediction:** The transformed data is fed into the saved SVM model to generate real-time predictions.

This approach ensures efficient and accurate predictions on new, unseen data by utilizing the computational efficiency and patterns learned by the pre-trained models.

---

## **Contact** :mailbox:
If you have any questions, need additional data, or have suggestions or feedback, feel free to contact me:

- :email: **Email:** venkatesht11000@gmail.com

---

**Thank You for Checking Out This Project!** :smile:
