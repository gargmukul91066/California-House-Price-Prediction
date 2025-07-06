<a id="readme-top"></a>

# House-Price-Predictor


## Overview

This project is a house price prediction system built using **Streamlit** and **scikit-learn**. It uses a linear regression model trained on the **USA Housing** dataset to estimate house prices based on factors such as area income, house age, number of rooms, number of bedrooms, and area population. The interactive web app allows users to input these parameters and receive real-time price predictions with an intuitive and clean interface.





### Learning Journey 🗺️

This project represents my venture into real estate analytics and machine learning. Here's my journey:

- **Inspiration:**  
  The volatile housing market inspired me to create a tool that could help people make informed decisions about real estate investments using data science.

- **Why I Made It:**  
  I wanted to build a practical application that demonstrates how machine learning can solve real-world problems while making complex predictions accessible to everyone through a simple interface.

- **Challenges Faced:**  
  - **Data Quality:** Ensuring the dataset was clean and representative of real market conditions.  
  - **Feature Selection:** Identifying the most impactful features for accurate price prediction.  
  - **Model Selection:** Balancing model complexity with prediction accuracy.  
  - **UI/UX Design:** Creating an interface that's both informative and easy to use.

- **What I Learned:**  
  - **Data Analysis:** Advanced techniques in data preprocessing and visualization.  
  - **Machine Learning:** How to implement and evaluate linear regression models effectively.  
  - **Web Development:** Building interactive web applications using Streamlit.  
  - **Statistical Analysis:** Gaining insights into correlations and feature importance in the context of real estate.

This journey has not only improved my technical skills but also deepened my understanding of real estate analytics and its potential to drive smart investment decisions.


<br>


## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Technologies Used](#technologies-used)
5. [Dataset](#dataset)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Results](#results)
9. [License](#license)
10. [Contact](#contact)

    
<br>

## Features🌟

- **Accurate Predictions:**  
  Leverages a trained linear regression model for reliable house price estimation.
- **Interactive UI:**  
  A user-friendly Streamlit app where you can enter housing details and get instant price predictions.
- **Visual Data Insights:**  
  Notebooks include data visualizations such as pair plots, distribution plots, and heatmaps to understand feature relationships.
- **Model Persistence:**  
  The trained model and scaler are saved as pickle files, ensuring quick deployment and reusability.
- **Clean Design:**  
  A modern, gradient-themed interface with clear input descriptions and an informative disclaimer.


<br>

## Installation🛠

1. **Clone the Repository:**
   ```bash
   https://github.com/gargmukul91066/USA-House-Price-Prediction.git
   cd USA-House-Price-Prediction
   ```

2. **Create & Activate a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

5. **(Optional) Use Dev Container:**
   - Open the project in an IDE that supports Dev Containers using `.devcontainer/devcontainer.json`.

<br>


## Usage🚀

### Running the Streamlit App

Launch the prediction system with:
```bash
streamlit run app.py
```

**Features include:**
- Input various house parameters
- Get instant price predictions
  
### Running the Jupyter Notebook

Explore the data analysis and model training:
```bash
jupyter notebook "USA House Price Prediction.ipynb"
```

<br>


## Technologies Used💻

- **Programming Language:**  
  - Python

- **Machine Learning:**  
  - scikit-learn (Linear Regression)

- **Data Handling & Visualization:**  
  - Pandas, NumPy, Matplotlib, Seaborn

- **Web Framework:**  
  - Streamlit

- **Model Persistence:**  
  - pickle

<br>



## Dataset📊

The project uses the `USA_Housing.csv` dataset which includes:

- **5,000+ house records**
- **6 key features:**
  - Average Area Income
  - Average House Age
  - Average Number of Rooms
  - Average Number of Bedrooms
  - Area Population
  - House Price (target variable)

**Key Statistics:**
- **Total Records:** 5,000+
- **Features:** 6
- **Target Variable:** House Price
- **Data Format:** CSV

<br>


## Data Preprocessing🔄

1. **Data Cleaning:**
   - Removal of null values
   - Handling of non-numeric entries
   - Outlier detection and treatment

2. **Feature Engineering:**
   - Correlation analysis
   - Feature importance ranking
   - Feature scaling and normalization

3. **Data Splitting:**
   - 60% training data
   - 40% testing data 

<br>


## Model Training🧠

- **Training Process:**  
  - A linear regression model is trained on selected features from the dataset.
- **Evaluation:**  
  - Model performance is evaluated using MAE, MSE, RMSE, and R² metrics.
- **Model Persistence:**  
  - The trained model is saved as `model.pkl` and the processed DataFrame as `df.pkl` for deployment.
    
<br>


## Results🏆

- The model achieved a reasonable prediction accuracy, with error metrics calculated as follows:
  
  - **MAE**: The average absolute difference between the predicted and actual prices.
    
  - **MSE** and **RMSE**: Measures of the model's accuracy, with RMSE providing insight into the spread of errors.
    
  - **R²**: Indicates the proportion of variance explained by the model.

<br>  




## Directory Structure📁

```plaintext
house-price-predictor/
├── README.md                    # Project documentation
├── USA House Price Prediction.ipynb # Jupyter notebook for analysis
├── USA_Housing.csv             # Dataset file
├── app.py                      # Streamlit application
├── df.pkl                      # Pickled DataFrame
├── model.pkl                   # Trained model file
└── requirements.txt            # Dependencies list
```

<br>






## Contact

### 📬 Get in Touch!
Feel free to reach out for collaborations or questions:

- [![GitHub](https://github.com/gargmukul91066) 💻 — Explore more projects by Mukul Garg.

<br>


## Thanks for exploring—happy predicting! 🏡

> "A home is more than just a structure—it's where dreams are built. Let data help you build your future." – Anonymous

<p align="right">
  (<a href="#readme-top">back to top</a>)
</p>
