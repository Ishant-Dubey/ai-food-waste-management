# AI-Based Food Waste Management

[![Live Demo](https://img.shields.io/badge/Live%20Demo-aifoodwaste.streamlit.app-FF4B4B.svg?logo=streamlit&logoColor=white)](https://aifoodwaste.streamlit.app)
[![Python Version](https://img.shields.io/badge/Python-3.13%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

> **Live Demo:** [https://aifoodwaste.streamlit.app](https://aifoodwaste.streamlit.app)

> [!NOTE]
> **Status:** This project is a work in progress and is not yet complete. The waste prediction model and core Streamlit UI are functional, while dashboard data tracking and donation advisory features are still under development.

---

## About the Project

An application that uses machine learning to estimate food waste quantities (in kg) for businesses like restaurants, cafeterias, cloud kitchens, and supermarkets based on inventory and environmental details.

### Features
- **Waste Prediction:** Estimates expected food waste using a trained Support Vector Regressor (SVR).
- **Cyclical Date Features:** Encodes day of the week and month using sine/cosine transformations.
- **Interactive UI:** Built using Streamlit for entering inventory parameters and viewing predictions.

---

## Dataset & Features

The model is trained on inventory records with the following input features:

| Feature | Type | Description |
|---|---|---|
| `Business_Type` | Categorical | Cafeteria, Cloud Kitchen, Restaurant, Supermarket |
| `Food_Category` | Categorical | Bakery, Dairy, Fruits, Meat, Prepared Food, Vegetables |
| `Purchase_Quantity` | Numeric | Total quantity purchased (kg / units) |
| `Daily_Demand` | Numeric | Expected daily customer demand |
| `Shelf_Life_Days` | Numeric | Shelf life in days |
| `Storage_Temperature` | Numeric | Storage temperature in °C |
| `Promotion` | Binary | Whether an active promotion is running (Yes/No) |
| `Weather` | Categorical | Sunny, Cloudy, Rainy, Hot |
| `Date` | Datetime | Transformed into cyclical `weekday` & `month` sin/cos features |
| **`Waste_Quantity`** | **Target (Numeric)** | **Recorded food waste in kg** |

---

## Model & Evaluation

A Support Vector Regressor (SVR) was tuned using 5-fold cross-validation (`GridSearchCV`).

- **Algorithm:** Support Vector Regressor (`SVR`)
- **Best Parameters:** `kernel='rbf'`, `C=100`, `gamma=0.01`, `epsilon=1.0`

### Performance Metrics

| Metric | Training Set | Testing Set |
|---|---|---|
| **$R^2$ Score** | **0.9546** | **0.9531** |
| **Mean Absolute Error (MAE)** | **1.50 kg** | **1.61 kg** |
| **Mean Squared Error (MSE)** | **11.88** | **14.32** |
| **5-Fold Cross-Validation $R^2$** | — | **0.9388** |

### Baseline Comparison

| Model | Test MAE | Test MSE | Test $R^2$ |
|---|---|---|---|
| Linear Regression | 4.19 kg | 59.98 | 0.8037 |
| K-Nearest Neighbors ($k=5$) | 3.59 kg | 46.96 | 0.8463 |
| Decision Tree Regressor | 1.70 kg | 26.44 | 0.9135 |
| Support Vector Regressor (Baseline) | 1.54 kg | 34.40 | 0.8874 |
| **Fine-Tuned SVR** | **1.61 kg** | **14.32** | **0.9531** |

---

## Project Structure

```
ai_food_waste_management/
├── dataset/
│   ├── food_waste_dataset.csv     # Raw dataset
│   ├── X_train.csv                # Training features (6,400 samples)
│   ├── X_test.csv                 # Testing features (1,600 samples)
│   ├── y_train.csv                # Training target
│   └── y_test.csv                 # Testing target
├── frontend/
│   └── app.py                     # Streamlit web application
├── models/
│   ├── SVR_Food_Waste.pkl         # Trained SVR model
│   ├── scaler.pkl                 # StandardScaler for numeric features
│   └── columns.pkl                # Feature column ordering
├── notebooks/
│   ├── food_waste_analysis.ipynb  # Data analysis and preprocessing
│   └── model_training.ipynb       # Model training and evaluation
├── pyproject.toml
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Ishant-Dubey/ai-food-waste-management.git
cd ai-food-waste-management
```

### 2. Set up environment and install dependencies

Using `uv`:
```bash
uv sync
```

Or using standard `venv` and `pip`:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -e .
```

### 3. Run the Streamlit app
```bash
streamlit run frontend/app.py
```
Open `http://localhost:8501` in your browser.
