# 🚚 Delivery ETA — Modélisation Prédictive des Temps de Livraison

A machine learning project that predicts **food delivery times** based on delivery, traffic, weather, vehicle, location, distance, and preparation-time information.

The project covers the complete machine learning workflow:

**Data Cleaning → EDA → Feature Engineering → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Prediction App**

---

## 📌 Project Overview

Accurately estimating delivery time is important for improving the customer experience and helping delivery platforms better understand the factors that affect delivery performance.

This project uses a food delivery dataset to build a regression model capable of predicting the expected delivery time in minutes.

The final model is an optimized **Random Forest Regressor** integrated into a Streamlit application.

---

## 🎯 Objectives

* Clean and prepare the raw delivery dataset
* Explore the main factors affecting delivery time
* Create useful features such as:

  * Preparation Time
  * Delivery Distance
* Compare several regression models
* Optimize the best-performing model using `RandomizedSearchCV`
* Evaluate the final model on unseen test data
* Build an interactive Streamlit application for predictions
* Visualize important patterns in the dataset

---

## 🗂️ Project Structure

```text
DeliveryETA-Mod-lisation-Pr-dictive-des-Temps-de-Livraison/
│
├── app/
│   ├── app.py
│   ├── prediction.py
│   ├── model_performance.py
│   └── visualization.py
│
├── data/
│   ├── food-delivery-6ab250c2ce296539694086.csv
│   └── cleaned_delivery_data.csv
│
├── models/
│   └── delivery_eta_model.joblib
│
├── README.md
└── ...
```

### Application files

| File                   | Description                               |
| ---------------------- | ----------------------------------------- |
| `app.py`               | Main Streamlit application and navigation |
| `prediction.py`        | Interactive delivery-time prediction      |
| `model_performance.py` | Displays model evaluation metrics         |
| `visualization.py`     | Displays exploratory data visualizations  |

---

## 📊 Dataset

The cleaned dataset contains:

* **41,953 rows**
* **21 columns**

Some of the main variables include:

* Delivery person's age
* Delivery person's rating
* Restaurant coordinates
* Delivery location coordinates
* Weather conditions
* Road traffic density
* Vehicle condition
* Type of order
* Type of vehicle
* Multiple deliveries
* Festival
* City
* Preparation time
* Delivery distance
* Delivery time

The target variable is:

```text
Time_taken
```

representing the delivery time in minutes.

---

## 🔧 Data Preparation

The dataset was cleaned and prepared before model training.

Main steps included:

* Handling missing values
* Cleaning numerical and categorical variables
* Converting date/time information
* Calculating preparation time
* Calculating delivery distance
* Removing unrealistic delivery times
* Preparing the target variable
* Encoding categorical features
* Splitting the dataset into training and testing sets

---

## 🧠 Feature Engineering

Two important features were created during the project.

### Preparation Time

The difference between:

```text
Time_Order_picked
-
Time_Orderd
```

was used to calculate:

```text
Preparation_Time_min
```

### Delivery Distance

The geographical coordinates of the restaurant and delivery location were used to calculate:

```text
Distance_km
```

These features provide the model with additional information about the delivery process.

---

## 🤖 Models Tested

Several regression algorithms were compared:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

The Random Forest model was then optimized using:

```python
RandomizedSearchCV
```

---

## ⚙️ Final Model

The final model is an optimized:

**Random Forest Regressor**

with the following main hyperparameters:

```text
n_estimators = 200
max_depth = 20
max_features = None
min_samples_split = 10
n_jobs = -1
random_state = 42
```

The complete preprocessing and model pipeline is stored in:

```text
models/delivery_eta_model.joblib
```

The model is stored using **Git LFS** because of its large file size.

---

## 📈 Model Performance

The final model was evaluated on unseen test data.

| Metric      |       Result |
| ----------- | -----------: |
| MAE         | **3.15 min** |
| RMSE        | **3.99 min** |
| R²          |    **0.815** |
| Adjusted R² |    **0.815** |

### Interpretation

The model has an average absolute prediction error of approximately:

**3.15 minutes**

on the test dataset.

The R² score of approximately **0.815** means that the model explains a substantial proportion of the variation in delivery time within this dataset.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit dashboard with three sections.

### 1. Prediction

Users can enter delivery information such as:

* Delivery person age
* Rating
* Vehicle condition
* Traffic
* Weather
* Order type
* Vehicle type
* Festival
* City
* Preparation time
* Distance
* Restaurant coordinates
* Delivery coordinates

The application then returns an estimated delivery time.

### 2. Model Performance

Displays:

* MAE
* RMSE
* R²
* Adjusted R²

### 3. Visualization

The application provides several visualizations:

* Delivery Time Distribution
* Distance vs Delivery Time
* Traffic Density vs Delivery Time
* Weather vs Delivery Time
* Preparation Time vs Delivery Time

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/DIRECT0R-ctrl/DeliveryETA-Mod-lisation-Pr-dictive-des-Temps-de-Livraison.git
```

Enter the project:

```bash
cd DeliveryETA-Mod-lisation-Pr-dictive-des-Temps-de-Livraison
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If the model is stored with Git LFS, make sure Git LFS is installed and initialized:

```bash
git lfs install
git lfs pull
```

---

## ▶️ Run the Application

From the project root:

```bash
python -m streamlit run app/app.py
```

Streamlit will start the application locally.

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Machine Learning

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* RandomizedSearchCV

### Application

* Streamlit

### Model Serialization

* Joblib

### Version Control

* Git
* GitHub
* Git LFS

---

## 📚 Machine Learning Concepts Practiced

This project was also used to practice several machine learning concepts:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Feature Selection
* Categorical Encoding
* Missing Value Imputation
* Standardization
* Train/Test Split
* Regression
* Model Evaluation
* Hyperparameters
* Hyperparameter Tuning
* Cross-Validation
* Overfitting
* Regularization concepts
* Model Pipelines

---

## 🔮 Possible Improvements

Future improvements could include:

* Feature importance visualization
* More advanced feature engineering
* Additional regression algorithms
* More extensive hyperparameter optimization
* Prediction confidence or uncertainty estimates
* Model monitoring
* Deployment to a cloud platform
* Automated data and model pipelines

---

## 👤 Author

**Aymane Laksimi**

AI / Data & Machine Learning Student

GitHub: `DIRECT0R-ctrl`

---

## 📄 License

This project was developed for educational and portfolio purposes.

