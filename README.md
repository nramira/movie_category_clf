# 🎬 Movie Categories Classifier

An end-to-end machine learning project that classifies Netflix movies and TV shows into multiple genres based on their description, cast, and director. Built using a modular ML pipeline and deployed through a lightweight Flask web app.

---

## 📌 Overview

This project demonstrates a full machine learning lifecycle, from data ingestion to deployment:

- Uses a **multilabel classification model**
- Based on metadata from Netflix (via Kaggle)
- Modular ML pipeline implemented with clear structure (`src/`)
- Text feature extraction using **TF-IDF**
- Model experimentation with multiple classifiers
- Deployed via a **Flask** web app for inference

---

## 📁 Project Structure

```bash
├── app.py # Flask application for inference
├── requirements.txt # Dependencies
├── setup.py # Package setup
├── README.md
├── /artifacts/ # Stores datasets, models, and transformers
│ ├── train.csv
│ ├── test.csv
│ ├── netflix_titles.csv
│ ├── model.pkl
│ └── preprocessor.pkl
├── /templates/ # HTML templates for Flask app
│ ├── index.html
│ └── home.html
├── /notebooks/ # Exploratory data analysis (EDA)
│ └── eda.ipynb
├── /src/ # Main ML package
│ ├── init.py
│ ├── logger.py
│ ├── exception.py
│ ├── utils.py
│ ├── /components/ # Data ingestion, transformation, training
│ │ ├── init.py
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ └── /pipeline/ # Training and prediction scripts
│ ├── init.py
│ ├── train_pipeline.py
│ └── predict_pipeline.py
```

---

## 📦 Dataset

- **Source**: [Netflix Movies and TV Shows Dataset on Kaggle](https://www.kaggle.com/datasets/anandshaw2001/netflix-movies-and-tv-shows)
- **Target Column**: `listed_in` (multilabel genres/categories)
- **Input Features**: `description`, `cast`, `director`

---

## 🧠 ML Pipeline

### 🔌 Data Ingestion

- Implemented in `data_ingestion.py`
- Connects to the **Kaggle API** to download the Netflix dataset automatically
- Splits the dataset into train and test CSVs
- Stores raw data in the `artifacts/` directory

### 🧹 Data Transformation

- Implemented in `data_transformation.py`
- Handles missing values, normalizes text and removes stop words
- Combines `description`, `cast`, and `director` into a single textual feature
- Uses `TfidfVectorizer` to transform text into numerical features
- Applies a multilabel binarizer to the target column
- Stores a preprocessing object to `preprocessor.pkl` for inference

### 🤖 Model Training

- Implemented in `model_trainer.py`
- Trains and evaluates multiple **MultiOutputClassifier** models:
  - **XGBoost**: `MultiOutputClassifier(XGBClassifier())`
  - **Multi-Layer Perceptron (MLP)**: `MultiOutputClassifier(MLPClassifier())`
  - **Extra Trees**: `MultiOutputClassifier(ExtraTreesClassifier())`
  - **K-Nearest Neighbors**: `MultiOutputClassifier(KNeighborsClassifier())`
- Logs performance metrics and saves the best model to `model.pkl`

---

## 🖥️ Flask Web App

- Located in `app.py`
- Accepts user input for `description`, `cast`, and `director`
- Returns predicted genres via web UI

---

## 💻 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-categories-clf.git
cd movie-categories-clf

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Ensure Kaggle API credentials are configured (~/.kaggle/kaggle.json)

# Train model
python src/pipeline/train_pipeline.py

# Run web app
python app.py
```

---

## 🧪 Results
The training pipeline prints evaluation metrics such as Hamming Loss and Jaccard Score. The best-performing model is persisted and used by the Flask app for predictions.