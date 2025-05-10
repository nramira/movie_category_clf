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

.
├── app.py # Flask application for inference
├── requirements.txt # Dependencies
├── setup.py # Package setup
├── .env # Environment variables (optional)
├── .gitignore
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