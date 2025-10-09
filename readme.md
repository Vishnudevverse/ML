# Advanced Car Price Predictor 🚗

This project is a web application built with Streamlit that uses a machine learning model to predict the price of used cars. It provides users with not just a price estimate but also valuable market insights to help them make informed decisions.

-----

## Features

  * **📈 Accurate Price Prediction:** Uses a `RandomForestRegressor` model to predict car prices based on features like brand, model year, mileage, and more.
  * **📊 Market Analysis:** Displays a price distribution plot for similar cars, showing users where their predicted price stands in the current market.
  * **🔥 "Good Deal" Identifier:** Analyzes similar car listings from the dataset and flags those that are priced significantly below their predicted market value.
  * **🧠 Explainable AI:** Shows a feature importance chart, revealing which factors (e.g., mileage, model year) have the biggest impact on the price.
  * **📋 Multi-Car Comparison:** Allows users to save multiple predictions to a comparison list for easy side-by-side analysis.
  * **🌍 Currency Conversion:** Correctly converts prices from the original USD dataset to INR for local relevance.

-----

## Technology Stack

  * **Backend:** Python
  * **Machine Learning:** Scikit-learn, Pandas, NumPy
  * **Web Framework:** Streamlit
  * **Plotting:** Matplotlib, Seaborn

-----

## Setup and Installation

Follow these steps to run the project on your local machine.

### **1. Prerequisites**

  * Python 3.8 or higher installed.
  * `pip` (Python package installer).

### **2. Clone the Repository (or Download Files)**

If you are using Git, clone the repository. Otherwise, just make sure all the project files (`app.py`, `train_model.py`, `used_cars.csv`) are in the same folder.

```bash
git clone https://github.com/Vishnudevverse/ML
cd ML
```

### **3. Train the Machine Learning Model**

Before you can run the app, you need to train the model. This will create the `.joblib` files.

```bash
python train_model.py
```

### **4. Run the Streamlit App**

Now, you're ready to launch the web application\!

```bash
streamlit run app.py
```

A new tab should open in your web browser with the running application.

-----

## Usage

1.  Use the controls in the **sidebar** on the left to enter the details of a car.
2.  Click the **"Predict Price"** button.
3.  View the predicted price, market analysis, and similar car listings on the main page.
4.  Click **"Add to Comparison"** to save a car's details to the comparison table at the bottom of the page.

-----

## Dataset

This project uses the "Used Car Price Prediction" dataset, which can be found on Kaggle. The dataset contains information on thousands of used car listings with various features.

*https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset?resource=download*