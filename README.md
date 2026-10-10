# Bank Deposit Prediction using Supervised Machine Learning

## 📌 Project Overview

This project uses supervised machine learning classification algorithms to predict whether a customer will subscribe to a bank deposit based on their personal information, financial details, and previous marketing campaign interactions.

The project explores multiple classification models, evaluates their performance, and applies hyperparameter tuning and ensemble learning techniques to improve predictive performance.

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on bank customer data.
- Handle categorical features using encoding techniques.
- Scale numerical features using StandardScaler.
- Train and evaluate multiple classification algorithms.
- Apply cross-validation and hyperparameter tuning.
- Explore ensemble learning techniques such as bagging, boosting, and stacking.
- Compare model performance using classification evaluation metrics.

## 📂 Dataset

The project uses a bank marketing dataset containing **11,162 records and 17 columns**.

The dataset includes information such as:

- `age` — Customer's age
- `job` — Type of occupation
- `marital` — Marital status
- `education` — Education level
- `balance` — Account balance
- `housing` — Housing loan status
- `loan` — Personal loan status
- `contact` — Contact communication type
- `duration` — Duration of the last contact
- `campaign` — Number of contacts during the current campaign
- `pdays` — Days since the customer was previously contacted
- `previous` — Number of contacts before the current campaign
- `poutcome` — Outcome of the previous marketing campaign
- `deposit` — Target variable indicating whether the customer subscribed to a deposit

The target variable is `deposit`, which contains two classes: `yes` and `no`.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## ⚙️ Project Workflow

### 1. Data Loading and Exploration
- Load the dataset using Pandas.
- Examine the dataset structure and data types.
- Check for missing values and duplicate records.
- Explore feature distributions and relationships.

### 2. Data Visualization
- Use box plots to inspect numerical features for potential outliers.
- Use a correlation heatmap to examine relationships between numerical features.

### 3. Data Preprocessing
- Apply Label Encoding to selected binary categorical features.
- Apply One-Hot Encoding to other categorical features.
- Split the dataset into training and testing sets.
- Standardize numerical features using StandardScaler.

### 4. Model Training and Evaluation

The following classification algorithms are explored:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)

Five-fold cross-validation is used to compare the initial models.

### 5. Hyperparameter Tuning

The project explores model optimization using:

- GridSearchCV
- RandomizedSearchCV

These techniques search for suitable hyperparameter combinations for selected classification algorithms.

### 6. Ensemble Learning

The following ensemble techniques are implemented:

- **Random Forest:** Combines multiple decision trees.
- **AdaBoost:** Builds an ensemble by focusing on previously misclassified observations.
- **Gradient Boosting:** Builds trees sequentially to improve predictive performance.
- **Stacking Classifier:** Combines predictions from multiple base models using a final estimator.
- **XGBoost:** Uses gradient boosting for classification.

### 7. Model Evaluation

Model predictions are evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Cross-validation scores

## 📁 Project Structure

```text
supervised_learning_on_classified_data/
│
├── bank.ipynb       # Data analysis, preprocessing, and model training
├── bank.csv         # Bank marketing dataset
└── README.md        # Project documentation
```

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/chandansingh-thakur/supervised_learning_on_classified_data.git
```

### Step 2: Navigate to the Project Directory

```bash
cd supervised_learning_on_classified_data
```

### Step 3: Install the Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `bank.ipynb` and execute the cells to explore the data, train the models, and evaluate their performance.

**Note:** The notebook currently loads the dataset using `pd.read_csv('bank.csv')`. Ensure the CSV file is named `bank.csv` and placed in the same directory as the notebook, or update the file path accordingly.

## 📈 Key Learnings

Through this project, I explored:

- Data preprocessing and feature encoding.
- Exploratory data analysis and visualization.
- Supervised learning classification algorithms.
- Cross-validation and hyperparameter optimization.
- Bagging, boosting, and stacking techniques.
- Model evaluation using classification metrics.

## 🔮 Future Improvements

- Compare all models using a consolidated performance table.
- Improve preprocessing using Scikit-learn pipelines.
- Use stratified cross-validation and address class imbalance where appropriate.
- Optimize decision thresholds based on precision and recall.
- Save the best-performing model and build a prediction interface.

## 👨‍💻 Author

**Chandan Singh**

B.Tech — Artificial Intelligence and Machine Learning

GitHub: [chandansingh-thakur](https://github.com/chandansingh-thakur)
