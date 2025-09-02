# Bank-Loan-Approval-Prediction
## 📌 Project Overview
This project focuses on building a machine learning model to predict personal loan approval based on customer demographic and financial information. Using the Bank Loan Dataset, we apply data preprocessing, exploratory data analysis (EDA), feature engineering, and neural network modeling to classify whether a customer is likely to be approved for a loan.

The notebook walks through different phases of model building, from initial overfitting experiments to final model evaluation and feature importance analysis.

## 🚀 Key Features
### Exploratory Data Analysis (EDA):
Summary statistics for each feature (min, max, mean, median, standard deviation).

Histograms of input features.

Visualization of class distribution.

### Data Preprocessing:
Normalization of numeric features.

Splitting into train and test sets.

### Modeling:
Multi-layer feedforward neural network built using TensorFlow/Keras.

Batch Normalization for stable training.

Binary classification with sigmoid activation.

### Evaluation:
Model accuracy measurement.

Comparison of multiple architectures.

Feature importance and dimensionality reduction.

## 🛠️ Technologies Used
Python

Pandas & NumPy for data handling

Matplotlib for visualization

TensorFlow / Keras for building and training neural networks

## 📂 Project Structure
├── Finalprojectcodefile.ipynb   # Main Jupyter Notebook with full code

├── bankloandatset.csv           # Dataset

├── ProjectReport.pdf            # An overleaf project report(documentation)

└── README.md

## 📊 Dataset
The dataset used is bankloandatset.csv.

It contains demographic and financial attributes of bank customers along with a target label.

Features: Age, Experience, Income, Family, CCAvg, Education, Mortgage, etc.

Target: Personal.Loan (0 = No, 1 = Yes)

⚠️ Make sure the dataset file is in the same directory as the notebook before running it.

## ⚡ How to Run
Clone this repository:

git clone https://github.com/your-username/bank-loan-prediction.git

cd bank-loan-prediction

Launch Jupyter Notebook:

jupyter notebook Finalprojectcodefile.ipynb

Run all cells to see EDA, model training, and results.

## 📈 Results
Achieved a final accuracy of ~95-98% (depending on hyperparameters).

Demonstrated importance of feature scaling and normalization.

Identified most relevant features contributing to loan approval.

## 🔮 Future Improvements
Implement cross-validation for more robust evaluation.

Compare with traditional ML models (Logistic Regression, Random Forest, XGBoost).

Deploy the model as a Flask API or Streamlit web app for real-time predictions.
