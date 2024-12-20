# Image-Tampering-Classification-with-Synthetic-Data-Augmentation

### **1. Project Overview** ✈ 👀
- **Title**: Image Tampering Classification with Synthetic Data Augmentation
- **Objective**: To classify images as authentic, real tampered, or synthetic tampered using a machine learning model with synthetic data augmentation.
- **Scope**: Includes data preprocessing, synthetic data generation, model training, and fine-tuning. Focused on image tampering detection.

---

### **2. Technologies Used** 💻
- **Programming Language**: Python
- **Frameworks and Libraries**: 
  - TensorFlow & Keras (Modeling)
  - Diffusers (Synthetic Image Generation)
  - Scikit-learn (Evaluation)
  - Seaborn & Matplotlib (Visualization)
  - PIL (Image Processing)
- **Platform**: Google Colab

---

### **3. Dataset** 🔢
- **Source**: CASIA2 Dataset
- **Structure**:
  - Authentic Images: Original, unaltered images
  - Tampered Images: Manually altered images
  - Synthetic Images: Generated using Stable Diffusion Inpainting
- **Balance Strategy**: Addressed class imbalance via oversampling.

---

### **4. Methodology** 🔎
1. **Data Preparation**:
   - Unzipped the dataset and organized into classes.
   - Generated synthetic tampered images using Stable Diffusion.
   - Balanced dataset through oversampling and downsampling.
   
2. **Model Architecture**:
   - Base Model: ResNet50 pre-trained on ImageNet.
   - Added custom layers:
     - `GlobalAveragePooling2D`
     - Fully connected layers
     - Output layer with softmax activation for 3 classes.

3. **Training**:
   - Initial training with frozen ResNet50 layers.
   - Fine-tuned by unfreezing the top layers.

4. **Evaluation Metrics**:
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Accuracy

---

### **5. Results** ⚓
#### **Initial Training**:
- **Accuracy**: 75%
- **Class Performance**:
  - Authentic: High precision and recall
  - Real Tampered: Poor recall
  - Synthetic Tampered: Very low precision and recall.

#### **Fine-Tuning**:
- Improved recall for Real Tampered images.
- Slight improvement in F1-scores for Synthetic Tampered images.

---

### **6. Limitations** 😶
- **Synthetic Data Impact**: The addition of synthetic images improved class distribution but showed limited impact on performance for real tampered images.
- **Imbalanced Class Performance**: Difficulty in accurately predicting tampered classes.

---
##Important Links:😃✨
CASIA2.zip: https://drive.google.com/file/d/1P23HDmXyXugG7uI9WqISEy9sxOrW6qfc/view?usp=sharing

Syntheic data: https://drive.google.com/drive/folders/1-BJ-yX2bw6mTBd2ow5SUvass2A4uuh29?usp=sharing

Kaggle Dataset: https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
