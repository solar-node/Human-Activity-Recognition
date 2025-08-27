# Human Activity Recognition Using Smartphones

This project demonstrates a complete machine learning workflow to classify human activities (like Walking, Sitting, Standing) based on smartphone sensor data. The final model, a highly-tuned XGBoost Classifier, achieves **94.16% accuracy** in predicting the user's activity.

This repository contains the full analysis in a Jupyter Notebook and a simple Python script to run a live prediction.

---

## 🌟 Features

- **Comprehensive Data Analysis:** The Jupyter Notebook (`human-activity-recognition.ipynb`) provides a detailed walkthrough of Exploratory Data Analysis (EDA), feature importance, and data preprocessing.
- **Multiple Model Comparison:** Implements and evaluates several models, including Logistic Regression, Random Forest, and a highly-tuned XGBoost classifier.
- **Predictive Script:** Includes a simple Python script (`predictActivity.py`) to load the final trained model and make a prediction on a random sample from the test data.
- **End-to-End Workflow:** Covers the entire data science lifecycle from data cleaning and visualization to model training, hyperparameter tuning, and final evaluation.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3 installed on your system. You will also need to install the libraries listed in the `requirements.txt` file.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/solar-node/Human-Activity-Recognition.git](https://github.com/solar-node/Human-Activity-Recognition.git)
    cd Human-Activity-Recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Prediction Script

This project includes a script to demonstrate how the final model works. It loads a pre-trained model and predicts the activity for a single random row from the `test.csv` dataset.

**Note:** To run this script, you will first need to download the four `.pkl` model files (`human_activity_model.pkl`, `scaler.pkl`, `pca.pkl`, `label_encoder.pkl`) and place them in the root of the project folder.

Execute the following command in your terminal:

```bash
python predictActivity.py
````

**Expected Output:**

```
Making a prediction for a random row (index: 1234) from test.csv

Prediction for row: 1234 -> The activity is WALKING
```

-----

## 📂 Project Structure

```
Human-Activity-Recognition/
│
├── Data/                   # Contains the training and test datasets.
│   ├── test.csv
│   └── train.csv
│
├── Models/                 # Stores the saved .pkl model files.
│   ├── HAR-model.pkl
│   ├── label_encoder.pkl
│   ├── pca.pkl
│   └── scaler.pkl
│
├── Plots/                  # Contains all generated charts and plots.
│   ├── Banner.png
│   ├── Confusion_Matrix_... (multiple files)
│   ├── Feature_Importance_plot.png
│   └── TSNE_visualization_of_all_features.png
│
├── .gitignore              # Tells Git which files to ignore.
│
├── human-activity-recognition.ipynb # Main notebook for analysis and modeling.
│
├── PredictActivity.py      # Script to run a sample prediction.
│
└── README.md               # Project documentation and user guide.
```

-----

## 🤖 Machine Learning Workflow

The core of this project is the Jupyter Notebook, which follows a structured approach to solving the classification problem:

1.  **Exploratory Data Analysis (EDA):** Investigated the dataset to understand the distribution of activities, identify key features, and visualize the separation between different classes using t-SNE.
2.  **Data Preprocessing:** Scaled the features using `StandardScaler` and encoded the target labels.
3.  **Dimensionality Reduction:** Applied Principal Component Analysis (PCA) to reduce the feature space while retaining 95% of the variance.
4.  **Model Training & Tuning:**
      - Trained baseline models (Logistic Regression, Random Forest, XGBoost).
      - Performed extensive hyperparameter tuning using `GridSearchCV` and `RandomizedSearchCV` to optimize all the models.
5.  **Evaluation:** The final tuned model was evaluated on the test set, achieving **94.16% accuracy**. The confusion matrix and classification report show high precision and recall across all six activity classes.

-----

## 🙏 Acknowledgments

This project uses the "Human Activity Recognition Using Smartphones" dataset from the UCI Machine Learning Repository.

  - **Source:** [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
  - **Creators:** Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
