# ML Classification Models - Credit Card Fraud Detection

## Problem Statement
This project implements and compares six different machine learning classification models to detect fraudulent credit card transactions. The goal is to identify which models perform best on highly imbalanced fraud detection data and deploy an interactive web application for model evaluation.

## Dataset Description
**Source:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Features:**
- **Total Features:** 30 (V1-V28 are PCA-transformed features, Time, Amount)
- **Target Variable:** Class (0 = Not Fraud, 1 = Fraud)
- **Total Instances:** 284,807 transactions
- **Class Distribution:** Highly imbalanced (0.17% fraudulent transactions)
- **Feature Types:** All numerical (continuous)

**Dataset Characteristics:**
- Time: Seconds elapsed between each transaction and the first transaction
- Amount: Transaction amount
- V1-V28: Principal components obtained via PCA transformation (original features not disclosed due to confidentiality)

## Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9991 | 0.9567 | 0.8505 | 0.6149 | 0.7137 | 0.7227 |
| Decision Tree | 0.9994 | 0.8423 | 0.8790 | 0.7365 | 0.8015 | 0.8043 |
| KNN | 0.9994 | 0.9188 | 0.9153 | 0.7297 | 0.8120 | 0.8170 |
| Naive Bayes | 0.9780 | 0.9552 | 0.0604 | 0.8041 | 0.1124 | 0.2168 |
| Random Forest (Ensemble) | 0.9995 | 0.9682 | 0.9569 | 0.7500 | 0.8409 | 0.8469 |
| XGBoost (Ensemble) | 0.9968 | 0.5255 | 0.0429 | 0.0405 | 0.0417 | 0.0401 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Good baseline performance with balanced precision (0.85) and recall (0.61). High AUC (0.96) indicates strong ability to distinguish between classes. Suitable for interpretable fraud detection. |
| Decision Tree | Strong precision (0.88) with improved recall (0.74) compared to logistic regression. Lower AUC (0.84) suggests some overfitting. Good interpretability for rule-based fraud patterns. |
| KNN | Best balance among traditional models with highest precision (0.92) and good recall (0.73). High AUC (0.92) shows robust performance. However, may be computationally expensive for large-scale deployment. |
| Naive Bayes | Severely underperformed despite high recall (0.80). Extremely low precision (0.06) causes too many false positives, making it impractical. The independence assumption doesn't hold for this dataset. |
| Random Forest (Ensemble) | **Best overall performer** with highest precision (0.96), strong recall (0.75), and best AUC (0.97). Excellent F1 (0.84) and MCC (0.85) scores indicate superior performance on imbalanced data. Recommended for deployment. |
| XGBoost (Ensemble) | Surprisingly poor performance across all metrics. Very low precision (0.04) and recall (0.04) suggest model configuration issues or class imbalance sensitivity. Requires hyperparameter tuning with scale_pos_weight adjustment for imbalanced data. |

## Installation & Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run app.py
```

## Streamlit App Features
-  Dataset upload functionality (CSV format)
-  Model selection dropdown (6 models)
-  Comprehensive evaluation metrics display
-  Interactive confusion matrix visualization
-  Detailed classification report

## Deployment
**Live App:** []
**GitHub Repository:** []

## Project Structure
```
ml-classification-streamlit/
│-- app.py                          
│-- requirements.txt                
│-- README.md                       
│-- model/
│   ├-- train_models.ipynb            
│   ├-- *.pkl                      
│   ├-- scaler.pkl                 
│   └-- model_results.csv          
│-- data/
│   ├-- creditcard.csv             # Original dataset
│   └-- test_data.csv              # Sample test data
```

## Key Findings
1. **Random Forest** achieved the best overall performance for fraud detection
2. **Class imbalance** significantly impacts model performance
3. **XGBoost requires tuning** with `scale_pos_weight` for imbalanced datasets
4. **Naive Bayes assumptions** don't hold for PCA-transformed features
5. **High precision is critical** to minimize false fraud alerts in production

## Author
Mrudhula Raya 
M.Tech (AI/ML) - BITS Pilani WILP
Machine Learning - Assignment 2