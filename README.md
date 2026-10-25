# ⭐ Intelligent Loan Risk Assessment System

> 🚀 An ML-based loan risk assessment system that analyzes applicant and financial data to predict loan approval outcomes and support faster, data-driven decision-making.

---

## 📌 Project Overview

The **Intelligent Loan Risk Assessment System** is a Machine Learning-based application developed to analyze loan application and financial-history data.

The system provides an end-to-end Machine Learning pipeline that includes:

- ⭐ Data integration and preprocessing
- ⭐ Missing-value and duplicate handling
- ⭐ Outlier handling
- ⭐ Feature engineering
- ⭐ Class imbalance handling using SMOTE
- ⭐ Machine Learning model training
- ⭐ Model evaluation and comparison
- ⭐ Loan approval prediction
- ⭐ Interactive Streamlit web interface

The system uses **Logistic Regression, Random Forest, and XGBoost** to identify patterns in financial data and predict loan approval outcomes.

> ⚠️ **Disclaimer:** Predictions are intended to support decision-making and should not replace the final decision of a financial institution. Model predictions require appropriate validation before being used in real lending decisions.

---

## 🎯 Project Objectives

- Integrate loan application and financial-history datasets.
- Clean and preprocess the collected data.
- Handle missing values, duplicates, outliers, and inconsistent data.
- Perform feature engineering to create meaningful financial indicators.
- Handle class imbalance using SMOTE or other appropriate resampling techniques.
- Train and compare Logistic Regression, Random Forest, and XGBoost.
- Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Select the best-performing model based on the chosen evaluation metric.
- Develop an interactive Streamlit web application for loan approval prediction.

---

## ❗ Problem Statement

Traditional loan evaluation processes can be time-consuming and may depend heavily on manual verification and fixed rules.

Loan datasets can contain several challenges, including:

- Missing values
- Duplicate records
- Outliers
- Inconsistent information
- Imbalanced target classes
- Complex relationships between financial attributes

Therefore, an automated Machine Learning-based system can help analyze financial information, identify patterns, and support the assessment of loan applications.

---

## 💡 Proposed Solution

The proposed system implements an end-to-end Machine Learning pipeline to process loan data, train classification models, evaluate their performance, and generate predictions through a Streamlit web application.

### 🔄 System Workflow

```text
       Loan Application Data
                 +
       Financial History Data
                 |
                 v
          Data Integration
                 |
                 v
           Data Cleaning
                 |
                 v
    Missing Value and Duplicate Handling
                 |
                 v
          Outlier Handling
                 |
                 v
         Feature Engineering
                 |
                 v
       Train-Test Data Splitting
                 |
                 v
       SMOTE / Resampling
       (Training Data Only)
                 |
                 v
           Model Training
                 |
                 v
          Model Evaluation
                 |
                 v
        Best Model Selection
                 |
                 v
      Loan Approval Prediction
                 |
                 v
       Streamlit Web Application
```

**Note:** Resampling techniques such as SMOTE must be applied only to the training data to prevent data leakage into the test set.

---

## 📊 Dataset Description

The project uses two main datasets.

### 1. Loan Application Dataset

Contains information related to loan applicants and their applications, such as:

- Customer details
- Applicant income
- Loan amount
- Credit-related information
- Loan application status

### 2. Financial History Dataset

Contains financial information related to applicants, such as:

- Savings
- Debt
- Previous loans
- Repayment information
- Customer financial history

### 🔗 Data Integration

The datasets are integrated using `Customer_ID`, which acts as the common identifier between the two datasets.

> **Dataset Note:** The project documentation describes 500 original records in each dataset, including duplicate rows used to demonstrate the data-cleaning process.

The datasets should be checked for matching identifiers, duplicate records, missing values, and data consistency before integration.

---

## ⚙️ Data Preprocessing

Data preprocessing improves data quality and prepares the dataset for Machine Learning.

### 1. Data Integration

Combines the loan application and financial-history datasets using `Customer_ID`.

### 2. Missing-Value Handling

Identifies missing values and handles them using appropriate imputation methods, depending on the data type and the preprocessing pipeline.

### 3. Duplicate Removal

Detects and removes unintended duplicate records to reduce redundancy.

### 4. Outlier Handling

Identifies unusual values in financial attributes and applies suitable handling techniques where necessary.

### 5. Data Consistency

Checks data types, invalid values, inconsistent categories, and other data-quality issues.

### 6. Feature Engineering

Creates additional financial indicators to improve the representation of applicant information.

| Feature | Description |
|---|---|
| **Total Income** | Represents the combined income of relevant income sources. |
| **Debt-to-Income Ratio** | Measures debt relative to income. |
| **Loan-to-Income Ratio** | Measures the requested loan amount relative to income. |
| **Disposable Income** | Estimates the income remaining after relevant financial obligations. |
| **Debt per Previous Loan** | Estimates average debt per previous loan, where applicable. |

**Example formulas:**

```text
Total Income = Applicant Income + Co-applicant Income

Debt-to-Income Ratio = Total Debt / Total Income

Loan-to-Income Ratio = Loan Amount / Total Income

Disposable Income = Total Income - Total Debt
```

These are illustrative formulas. The actual calculations must match the columns, units, and definitions used in the project datasets. Division-by-zero and missing-value cases should also be handled.

### 7. Class Imbalance Handling

SMOTE or another suitable resampling technique can be used to address class imbalance.

Resampling should be performed only on the training data, preferably within the Machine Learning pipeline or within each cross-validation training fold.

---

## 🤖 Machine Learning Models

The system considers three Machine Learning classification algorithms.

| Model | Purpose |
|---|---|
| **Logistic Regression** | Provides a baseline classification model and estimates class probabilities. |
| **Random Forest** | Combines multiple decision trees to capture nonlinear relationships and can provide feature-importance estimates. |
| **XGBoost** | Uses gradient boosting to learn complex patterns in structured data. |

### Model Training Process

1. Prepare the cleaned and engineered dataset.
2. Separate input features and the target variable.
3. Split the data into training and testing sets.
4. Apply preprocessing and resampling to the training data.
5. Train the candidate classification models.
6. Evaluate each model using appropriate metrics.
7. Compare the evaluation results.
8. Select the best-performing model according to the chosen selection criterion.

---

## 📈 Model Evaluation

The system evaluates the candidate models using multiple performance metrics.

| Metric | Description |
|---|---|
| **Accuracy** | Measures the proportion of all predictions that are correct. |
| **Precision** | Measures the proportion of positive predictions that are correct. |
| **Recall** | Measures the proportion of actual positive cases identified correctly. |
| **F1-Score** | Combines Precision and Recall using their harmonic mean. |
| **ROC-AUC** | Measures how well the model distinguishes between the two classes across classification thresholds. |
| **Cross-Validation** | Evaluates model performance across multiple training and validation folds. |

### Evaluation Configuration

The project uses the following configuration for model evaluation:

- **Train-Test Split:** 80:20
- **Stratification:** Used to preserve class proportions in the train and test sets.
- **Random State:** `42`
- **Model Selection Criterion:** Highest F1-Score among the candidate models.

The selected model should also be reviewed using Recall, ROC-AUC, and other relevant metrics, particularly because incorrect loan-risk predictions can have financial consequences.

> **Important:** The best model and its performance should be reported using actual experimental results. No performance scores are assumed or fabricated in this README.

---

## 🖥️ Streamlit Web Application

The project includes an interactive web application built using Streamlit.

### ✨ Main Features

- User-friendly prediction interface
- Loan applicant information input
- Financial information analysis
- Machine Learning-based prediction
- Saved preprocessing pipeline
- XGBoost model integration
- Prediction results displayed through a web interface

### Application Workflow

1. Open the Streamlit application.
2. Enter the required applicant and financial information.
3. Submit the input details.
4. The application processes the input using the saved preprocessing pipeline.
5. The trained XGBoost model generates a prediction.
6. The application displays the prediction result.

The preprocessing pipeline and model must use compatible feature names, data types, and transformations during both training and inference.

---

## 🗂️ Project Structure

```text
Intelligent_Loan_Risk_Website/
│
├── app.py
├── preprocessor.pkl
├── xgb_model.json
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Contains the Streamlit web application. |
| `preprocessor.pkl` | Stores the fitted preprocessing pipeline, if saved in this format. |
| `xgb_model.json` | Stores the trained XGBoost model. |
| `requirements.txt` | Lists the Python dependencies required to run the application. |
| `.gitignore` | Specifies files and directories that Git should ignore. |
| `README.md` | Provides project documentation and setup instructions. |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Preprocessing, classification, and model evaluation |
| **XGBoost** | Gradient-boosting classification |
| **Imbalanced-learn** | SMOTE and class-imbalance handling |
| **Matplotlib** | Data visualization |
| **Seaborn** | Exploratory data analysis and statistical visualization |
| **Joblib** | Saving and loading compatible Python objects |
| **Streamlit** | Interactive web application |

---

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

Install the following:

- Python
- Git
- A code editor such as Visual Studio Code

### 1. Clone the Repository

Open a terminal or PowerShell window and run:

```bash
git clone https://github.com/Amruthavalli-12/Intelligent_Loan_Risk_Website.git
```

Navigate to the project directory:

```bash
cd Intelligent_Loan_Risk_Website
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**macOS or Linux:**

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

After starting the application, Streamlit will display a local URL in the terminal. Open that URL in your web browser to access the application.

### Troubleshooting

- Verify that Python is installed and available in your system's PATH.
- Ensure that the virtual environment is activated.
- Check that `requirements.txt` contains the required dependencies.
- Confirm that `preprocessor.pkl` and `xgb_model.json` exist at the paths expected by `app.py`.
- Ensure that the installed XGBoost and scikit-learn versions are compatible with the saved model and preprocessing artifacts.

---

## 🔮 Future Enhancements

The project can be extended with the following improvements:

- Hyperparameter tuning
- Explainable AI using feature-contribution methods
- Cloud deployment
- Secure database integration
- Automated report generation
- Periodic model retraining
- Model performance monitoring
- Updated financial information
- Improved user interface
- Improved input validation and error handling
- Fairness evaluation and bias monitoring
- Prediction confidence and appropriate explanations

---

## ⚠️ Responsible Use

Loan-related predictions can affect access to financial services. Therefore:

- Predictions should be treated as decision-support information rather than definitive financial decisions.
- The model should be evaluated for bias and performance differences across relevant groups.
- Input data should be handled securely and in accordance with applicable privacy requirements.
- The system should be validated on representative data before any real-world use.
- Final lending decisions should follow the institution's policies and applicable laws and regulations.

---

## ✅ Conclusion

The **Intelligent Loan Risk Assessment System** demonstrates an end-to-end Machine Learning approach to analyzing loan application and financial-history data.

By combining data integration, preprocessing, feature engineering, class-imbalance handling, multiple classification algorithms, model evaluation, and a Streamlit web interface, the project provides a practical framework for predicting loan approval outcomes.

The system demonstrates how Machine Learning can support faster, more consistent, and data-driven loan assessment while emphasizing the importance of model validation, responsible use, and human oversight.

---

## 👩‍💻 Team

**Department of Artificial Intelligence & Machine Learning**  
**Academic Batch: 2024–2028**

| S. No. | Team Member | Student ID |
|---|---|---|
| 1 | M. Amruthavalli Supriya | `24B11AI239` |
| 2 | M. Padmavathi | `24B11AI276` |
| 3 | N. Shanmukha Simhadri Devi | `24B11AI281` |
| 4 | P. Varshitha Devi | `24B11AI210` |

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## ⭐ Support the Project

If you find this project useful, please consider giving the repository a ⭐ **Star** on GitHub!

**Built with Python, Machine Learning, and Streamlit.** 🚀
