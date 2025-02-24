# Wild Animal Classification: Endangered vs. Non-Endangered

This project focuses on classifying wild animals as **endangered** or **non-endangered** using **machine learning models**. It explores the implementation of **Convolutional Neural Networks (CNNs)** and **classical machine learning algorithms** (e.g., Logistic Regression, Random Forest) while applying optimization techniques such as **regularization**, **dropout**, and **early stopping**. The goal is to improve model performance, convergence speed, and efficiency.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview
The project aims to:
- Classify wild animals as **endangered** or **non-endangered** using image data.
- Compare the performance of **neural networks** with **classical machine learning models**.
- Apply optimization techniques like **regularization**, **dropout**, and **early stopping** to improve model performance.
- Evaluate models using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.

---

## Dataset
The dataset consists of images of wild animals categorized into two classes:
- **Endangered**: Animals at risk of extinction.
- **Non-Endangered**: Animals not currently at risk.

### Dataset Details
- **Source**: Publicly available dataset (e.g., Kaggle, wildlife conservation organizations).
- **Size**: Approximately 1,000 images (500 per class).
- **Format**: Images are resized to 150x150 pixels and normalized to the range [0, 1].
- **Split**:
  - **Training**: 60% of the data.
  - **Validation**: 20% of the data.
  - **Test**: 20% of the data.

---

## Methodology
### 1. Data Preprocessing
- Images are resized to 150x150 pixels.
- Pixel values are normalized to the range [0, 1].
- Data is split into training, validation, and test sets.

### 2. Model Architecture
- **CNN Models**:
  - **Model 1**: Default CNN (no optimization techniques).
  - **Model 2**: CNN with L2 Regularization.
  - **Model 3**: CNN with L2 Regularization, Adam optimizer, and early stopping.
  - **Model 4**: CNN with L1 Regularization and RMSprop optimizer.
  - **Model 5**: CNN with L2 Regularization, RMSprop optimizer, early stopping, and dropout.
- **Classical ML Models**:
  - **Logistic Regression**: Tuned using GridSearchCV.
  - **Random Forest**: Tuned using GridSearchCV.

### 3. Optimization Techniques
- **Regularization**: L1 and L2 regularization to prevent overfitting.
- **Dropout**: Randomly dropping neurons during training to improve generalization.
- **Early Stopping**: Stopping training when validation loss stops improving.
- **Hyperparameter Tuning**: Using GridSearchCV for classical ML models.

### 4. Evaluation Metrics
- **Accuracy**: Proportion of correctly classified samples.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the ROC curve.

---
## Link to my Demo Video:
https://youtu.be/fYw_hd9Avn0


## Results
### Model Performance
| Model                                      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------------------------------|----------|-----------|--------|----------|---------|
| Vanilla Model (No Optimization)            | 0.7904   | 0.83      | 0.82   | 0.82     | 0.91    |
| L2 Regularization with Adam                | 0.6854   | 0.85      | 0.84   | 0.84     | 0.92    |
| L2 Reg with Adam and Early Stopping        | 0.4579   | 0.86      | 0.85   | 0.85     | 0.93    |
| L1 Regularization with RMSprop             | 0.4579   | 0.84      | 0.83   | 0.83     | 0.92    |
| L2 Reg with RMSprop, Early Stopping, Dropout | 0.4579 | 0.87      | 0.86   | 0.86     | 0.94    |

### Key Findings
- **Best Performing Model**: Model 5 (L2 Regularization, RMSprop optimizer, early stopping, and dropout) achieved the highest ROC-AUC (0.94).
- **Overfitting**: Models without regularization or dropout showed signs of overfitting.
- **Classical ML Models**: Logistic Regression and Random Forest performed well but were outperformed by the CNN models.

---

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/wild-animal-classification.git
   cd wild-animal-classification
