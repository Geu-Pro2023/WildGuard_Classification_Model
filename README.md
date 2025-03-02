## **WildGuard Classification Model: Endangered vs. Non-Endangered Wild Animals**  

### **Project Overview**  
The WildGuard Classification Model aims to classify wild animals as endangered or non-endangered using machine learning techniques. This project employs Convolutional Neural Networks (CNNs) and classical machine learning algorithms, such as Logistic Regression and Random Forest, while applying various optimization techniques to enhance model performance, convergence speed, and efficiency. It also aims to:  

- Apply optimization techniques like regularization, dropout, and early stopping to improve model performance.  
- Evaluate models using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.  

### **Problem Statement**  
Wildlife conservation is critical, and identifying endangered species is essential for implementing effective conservation strategies. This project addresses the challenge of classifying images of wild animals into two categories: endangered and non-endangered. By leveraging machine learning, we aim to automate this classification process, providing a tool to assist conservationists and researchers.  

### **Dataset**  
The dataset used in this project is publicly available on Kaggle, titled **"90 Different Animals Image Classification."** It consists of approximately **1,000 images** (**600 images used in this project, with 300 per class**) of various wild animals, categorized into two classes: **Endangered** and **Non-Endangered**. The images are resized to **150x150 pixels** and normalized to the range **[0, 1]**.  

### **Methodology**  

#### **1. Data Preprocessing**  
- Images are resized to **150x150 pixels**.  
- Pixel values are normalized to the range **[0, 1]**.  
- Data is split into **training, validation, and test sets**.  

#### **2. Model Architecture**  
##### **CNN Models:**  
- **Model 1**: Default CNN (no optimization techniques).  
- **Model 2**: CNN with L2 Regularization.  
- **Model 3**: CNN with L2 Regularization, Adam optimizer, and early stopping.  
- **Model 4**: CNN with L1 Regularization and RMSprop optimizer.  
- **Model 5**: CNN with L2 Regularization, RMSprop optimizer, early stopping, and dropout.  

##### **Classical ML Models:**  
- **Logistic Regression**: Tuned using GridSearchCV.  
- **Random Forest**: Tuned using GridSearchCV.  

---

## **Table: Training Instances with Optimization Techniques and Metrics**  

| Instance  | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | F1-Score | Precision | Recall | Loss  |
|-----------|----------|-------------|--------|---------------|--------|--------------|----------|---------|-----------|--------|------|
| **1 (Default CNN)** | Adam | None | 20 | No | 3 Conv + 2 Dense | 0.001 | **73%** | 0.73 | 0.74 | 0.74 | 1.54 |
| **2 (L2 + Adam)** | Adam | L2 (λ=0.001) | 20 | No | 3 Conv + 2 Dense | 0.001 | **70%** | 0.70 | 0.70 | 0.70 | 1.14 |
| **3 (L2 + EarlyStop)** | Adam | L2 (λ=0.01) | 5* | Yes (Patience=4) | 3 Conv + 2 Dense | 0.001 | **53%** | 0.36 | 0.53 | 0.53 | 1.41 |
| **4 (L1 + RMSprop)** | RMSprop | L1 (λ=0.01) | 20 | No | 3 Conv + 2 Dense | 0.001 | **53%** | 0.36 | 0.53 | 0.53 | 0.72 |
| **5 (L2 + Dropout)** | RMSprop | L2 (λ=0.01) | 4* | Yes (Patience=4) | 3 Conv + 2 Dense | 0.001 | **48%** | 0.31 | 0.47 | 0.48 | 1.04 |

---

## **Error Analysis and Justification of Results**  

### **Default CNN (No Optimization)**  
- **Accuracy:** **73%**, **Loss:** **1.54**  
- **Issue:** Overfitting—training accuracy reached **100%**, but validation plateaued at ~**60%**.  
- **Cause:** No regularization or dropout, leading to memorization of noise.  

### **L2 Regularization (λ=0.001)**  
- **Accuracy:** **70%**, **Loss:** **1.14**  
- **Improvement:** Reduced overfitting; validation loss dropped by **25%**.  
- **Trade-off:** Suppressed model flexibility, worsening precision/recall balance.  

### **L2 + Early Stopping**  
- **Accuracy:** **53%**, **Loss:** **1.41**  
- **Issue:** Training stopped early (**epoch 5**), preventing learning.  
- **Cause:** High L2 (**λ=0.01**) and premature stopping discarded useful features.  

### **L1 Regularization + RMSprop**  
- **Accuracy:** **53%**, **Loss:** **0.72**  
- **Issue:** Excessive sparsity—L1 (**λ=0.01**) removed critical filters.  
- **Optimizer Impact:** RMSprop struggled with sparse gradients.  

### **L2 + Dropout + Early Stopping**  
- **Accuracy:** **48%**, **Loss:** **1.04**  
- **Issue:** Over-Dropout (**50%**) disrupted learning; early stopping (**epoch 4**) prevented recovery.  

---

## **Key Takeaways: Summary of Optimization Impact**  
- **L2 regularization reduces overfitting** by penalizing large weights, improving generalization. However, excessive L2 strength (**λ=0.01**) can suppress learning, leading to underperformance.  
- **L1 regularization harms small datasets** due to feature loss.  
- **Overly aggressive early stopping leads to underfitting.**  
- **Adam optimizer outperformed RMSprop** by handling noisy gradients better.  
- **50% dropout was too high**—**20-30%** might be optimal.  

### **Practical Implications for Conservation**  
The results indicate that classical ML models like **Random Forest** may be more effective for endangered species classification when working with **limited image data**. This insight is valuable for conservationists who may lack large-scale datasets but need reliable AI tools for decision-making. **Future improvements, such as transfer learning with ResNet50, could bridge the gap and enhance real-world deployment.**  

---

## **Classical Machine Learning Models**  
In addition to the CNN models, the best classical machine learning models were implemented:  

**Random Forest:**  
- **Accuracy:** **75%**  
- **Precision:** **0.88**  
- **Recall:** **0.64**  
- **F1-Score:** **0.74**  

### **Best Model**  
- **Accuracy: 76% (Test Set)**  
- **Why It Performed Best:**  
  - Handled image features better after flattening.  
  - Less prone to overfitting compared to CNNs.  
  - Effective with **limited data (600 images)**.  

---

## **Installation**  
To run this project locally, follow these steps:  

1. **Clone the repository:**  
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

## Link to my Demo Video:
https://youtu.be/fYw_hd9Avn0
