# Wild Animal Classification: Endangered vs. Non-Endangered

## Project Overview
The WildGuard Classification Model aims to classify wild animals as endangered or non-endangered using machine learning techniques. This project employs Convolutional Neural Networks (CNNs) and classical machine learning algorithms, such as Logistic Regression and Random Forest, while applying various optimization techniques to enhance model performance, convergence speed, and efficiency.
It also aims to:
- Apply optimization techniques like **regularization**, **dropout**, and **early stopping** to improve model performance.
- Evaluate models using metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.

---
## Problem Statement
Wildlife conservation is critical, and identifying endangered species is essential for implementing effective conservation strategies. This project addresses the challenge of classifying images of wild animals into two categories: endangered and non-endangered. By leveraging machine learning, we aim to automate this classification process, providing a tool to assist conservationists and researchers.

## Dataset
The dataset used in this project is publicly available on Kaggle, titled **"90 Different Animals Image Classification."** It consists of approximately 1,000 images (600 images used in this project, with 300 per class) of various wild animals, categorized into two classes: Endangered and Non-Endangered. The images are resized to 150x150 pixels and normalized to the range [0, 1].

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

### 3. Neural Network Optimizations
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
## Results
| Instance                | Optimizer | Regularizer      | Epochs | Early Stopping          | Layers             | Learning Rate | Accuracy | F1-Score | Precision | Recall | Loss |
|-------------------------|-----------|-----------------|--------|------------------------|--------------------|--------------|----------|----------|-----------|--------|------|
| 1 (Default CNN)        | Adam      | None            | 20     | No                     | 3 Conv + 2 Dense  | 0.001        | 73%      | 0.73     | 0.74      | 0.74   | 1.54 |
| 2 (L2 + Adam)         | Adam      | L2 (λ=0.001)    | 20     | No                     | 3 Conv + 2 Dense  | 0.001        | 70%      | 0.70     | 0.70      | 0.70   | 1.14 |
| 3 (L2 + EarlyStop)    | Adam      | L2 (λ=0.01)     | 5*     | Yes (Patience=4)       | 3 Conv + 2 Dense  | 0.001        | 53%      | 0.36     | 0.53      | 0.53   | 1.41 |
| 4 (L1 + RMSprop)      | RMSprop   | L1 (λ=0.01)     | 20     | No                     | 3 Conv + 2 Dense  | 0.001        | 53%      | 0.36     | 0.53      | 0.53   | 0.72 |
| 5 (L2 + Dropout)      | RMSprop   | L2 (λ=0.01)     | 4*     | Yes (Patience=4)       | 3 Conv + 2 Dense  | 0.001        | 48%      | 0.31     | 0.47      | 0.48   | 1.04 |


### Model Performance
| Model                                      | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------------------------------------------|----------|-----------|--------|----------|---------|
| Model1 (Default CNN)                       | 0.7904   | 0.83      | 0.82   | 0.82     | 0.91    |
| Model2 (L2 Regularization)                 | 0.6854   | 0.85      | 0.84   | 0.84     | 0.92    |
| Model3 (L2 Regularization, Early Stopping) | 0.4579   | 0.86      | 0.85   | 0.85     | 0.93    |
| Model4 (L1 Regularization, RMSprop)        | 0.4579   | 0.84      | 0.83   | 0.83     | 0.92    |
| Model5 (L2 Regularization,RMSprop, ES, Dp) | 0.4579   | 0.87      | 0.86   | 0.86     | 0.94    |


## Classical Machine Learning Models
In addition to the CNN models, the best classical machine learning models was implemented:

- **Random Forest:**
Accuracy: 0.75
Precision: 0.88
Recall: 0.64
F1-Score: 0.74


### Key Findings
- **Best Performing Model**: Model 5 (L2 Regularization, RMSprop optimizer, early stopping, and dropout) achieved the highest ROC-AUC (0.94).
- **Overfitting**: Models without regularization or dropout showed signs of overfitting.
- **Classical ML Models**: Logistic Regression and Random Forest performed well but were outperformed by the CNN models.

---
## Best Model
**Accuracy:** 76% (Test Set)
Why It Performed Best: It handled image features better after flattening and was less prone to overfitting compared to CNNs. and Effective with limited data (600 images).

## Link to my Demo Video:
https://youtu.be/fYw_hd9Avn0


## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/wild-animal-classification.git
   cd wild-animal-classification

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Make Predictions:**
   ```sh
   from utils import predict_and_display
   predict_and_display("4f98c92165.jpg")  # Replace with your image
   ```

## Conclusion
- **CNN Challenges:** All CNN variants overfit despite optimizations, likely due to limited training data.
- **Classical Models:** Random Forest outperformed CNNs, achieving 76% accuracy with better generalization.
- **Future Work:** Data augmentation, transfer learning (e.g., ResNet50), and larger datasets.
