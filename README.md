

# Iris Classification Model

This project demonstrates how to build and evaluate multiple classification models on the **Iris dataset**, a well-known dataset for machine learning.
The goal is to predict the species of Iris flowers based on four features: sepal length, sepal width, petal length, and petal width. 
The following machine learning algorithms are implemented and evaluated:

- Logistic Regression (LR)
- Linear Discriminant Analysis (LDA)
- k-Nearest Neighbors (KNN)
- Classification and Regression Tree (CART)
- Naive Bayes (NB)
- Support Vector Machine (SVM)

## Libraries Used

- **Python 3.x**: Programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For handling arrays and numerical operations.
- **Matplotlib**: For plotting graphs and visualizations.
- **Scikit-learn**: For implementing machine learning models, model selection, and evaluation.

## Dataset

The **Iris dataset** is loaded from the UCI Machine Learning Repository. It contains 150 samples of Iris flowers, divided into three species:

1. **Iris-setosa** (50 samples)
2. **Iris-versicolor** (50 samples)
3. **Iris-virginica** (50 samples)

Each sample has four features: 
- sepal length
- sepal width
- petal length
- petal width

The dataset is used for classification to predict the species of Iris flowers based on these features.

## Project Workflow

1. **Data Loading**: The dataset is loaded from a URL into a Pandas DataFrame.
2. **Data Exploration**: The shape and first few rows of the dataset are printed to verify data.
3. **Descriptive Statistics**: The dataset's summary statistics (mean, std, min, max) are displayed.
4. **Data Visualization**: Various plots are created to visualize the dataset:
   - **Boxplot**: To show the distribution of each feature.
   - **Histogram**: To visualize the distribution of the features.
   - **Scatter Matrix**: To visualize the relationships between features.
5. **Data Preprocessing**: The dataset is split into input features (X) and target labels (Y). Then, the data is split into training and testing sets (80% train, 20% test).
6. **Model Training and Evaluation**: Six different machine learning algorithms are evaluated using k-fold cross-validation (10 folds):
   - Logistic Regression (LR)
   - Linear Discriminant Analysis (LDA)
   - k-Nearest Neighbors (KNN)
   - Classification and Regression Tree (CART)
   - Naive Bayes (NB)
   - Support Vector Machine (SVM)
7. **Performance Metrics**: The accuracy and standard deviation of each model are calculated and displayed.

## Results

The models' performance (accuracy and standard deviation) is as follows:

- **Logistic Regression (LR)**: 95.83% (±4.17%)
- **Linear Discriminant Analysis (LDA)**: 97.50% (±3.82%)
- **k-Nearest Neighbors (KNN)**: 95.83% (±4.17%)
- **Classification and Regression Tree (CART)**: 95.00% (±4.08%)
- **Naive Bayes (NB)**: 96.67% (±4.08%)
- **Support Vector Machine (SVM)**: 94.17% (±5.34%)

## Conclusion

- **LDA** performed the best in terms of accuracy (97.5%), followed closely by **Logistic Regression (95.83%)** and **k-Nearest Neighbors (95.83%)**.
- **Support Vector Machine (SVM)** performed the least well among the models tested.

## Installation and Usage

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/iris-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd iris-classification
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Run the notebook (`iris_classification.ipynb`) to see the analysis and results.

## Dependencies

- `scipy==1.13.1`
- `numpy==1.26.4`
- `matplotlib==3.9.2`
- `pandas==2.2.2`
- `scikit-learn==1.5.1`
- `python==3.12.7`

