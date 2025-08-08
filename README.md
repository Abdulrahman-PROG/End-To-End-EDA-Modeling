# End-To-End-EDA-Modeling
Forecasting Rice Production in Sumatra (1993–2020)
Overview
This project focuses on forecasting rice production across eight provinces on Sumatra Island (Nanggroe Aceh Darussalam, North Sumatra, West Sumatra, Riau, Jambi, South Sumatra, Bengkulu, and Lampung) over the period from 1993 to 2020. Sumatra, where over 50% of land in each province is dedicated to agriculture, primarily produces rice as the dominant food commodity, alongside minor crops like corn, peanuts, and sweet potatoes. This project analyzes the impact of climate change on rice production, considering factors such as temperature rise, altered rainfall patterns, evaporation, water runoff, soil moisture, and climate variability, which affect planting patterns, timing, and crop yield and quality.
The methodology and findings are inspired by approaches like those in the Kaggle notebook on EDA and modeling and research on rice production forecasting using machine learning techniques.
Features

Exploratory Data Analysis (EDA): Analyzes historical rice production data alongside climate variables to identify trends, correlations, and anomalies.
Data Preprocessing: Cleans and prepares 28 years of data (1993–2020) for modeling, addressing missing values and standardizing formats.
Feature Engineering: Incorporates climate-related features (e.g., temperature, rainfall, soil moisture) to capture their impact on rice production.
Machine Learning Forecasting: Implements models like Linear Regression, Random Forest, Gradient Boosting, SVR, KNN Regression, and Decision Tree Regression to predict rice production.
Climate Impact Assessment: Evaluates how climate change factors (e.g., El Niño, La Niña, temperature rise) affect rice yields and agricultural sustainability.
Visualization: Uses Matplotlib and Seaborn to create visualizations such as time-series plots, heatmaps, and correlation matrices.

Requirements
To run this project, you need the following dependencies:

Python 3.6+
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

Install the required packages using:
pip install pandas numpy matplotlib seaborn scikit-learn

Dataset
The dataset covers 1993–2020 and includes:

Rice Production Data: Annual rice yields across eight Sumatra provinces (Nanggroe Aceh Darussalam, North Sumatra, West Sumatra, Riau, Jambi, South Sumatra, Bengkulu, Lampung).
Climate Variables: Temperature, rainfall, evaporation, water runoff, soil moisture, and climate variability metrics (e.g., El Niño/La Niña events).
Source: Aggregated from regional agricultural records and climate data sources (e.g., Statistics Indonesia, Ministry of Agriculture).

Usage

Clone or Download: Clone this repository or download the code and dataset.
Prepare Dataset: Place the dataset in the project directory and update the file path in the script.
Run the Notebook:
Open the notebook in Jupyter or a compatible environment.
Execute cells sequentially to perform EDA, preprocess data, train models, and generate forecasts.


Output: The project produces:
Visualizations of trends and climate impacts.
Forecasted rice production for each province.
Model performance metrics (e.g., RMSE, R²) for Linear Regression and other algorithms.



How It Works

Data Loading: Import rice production and climate data using Pandas.
EDA: Analyze distributions, trends, and correlations between rice yields and climate variables (e.g., rainfall, temperature) using visualizations.
Preprocessing: Handle missing values, encode categorical variables (e.g., provinces), and scale numerical features.
Feature Engineering: Create features like seasonal climate indices or lagged production values to capture temporal and climate effects.
Modeling: Train machine learning models (e.g., Linear Regression, Random Forest) to forecast rice production, comparing their performance.
Evaluation: Assess models using metrics like RMSE and R², and analyze the impact of climate variables on predictions.
Climate Impact Analysis: Quantify how factors like El Niño (drought) and La Niña (flood) affect rice yields, referencing studies like Murniati and Mutolib (2020).

Example
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("sumatra_rice_climate_1993_2020.csv")

# EDA: Visualize rice production trends
sns.lineplot(x='year', y='rice_production', hue='province', data=data)
plt.show()

# Train Linear Regression model
X = data[['temperature', 'rainfall', 'soil_moisture']]
y = data['rice_production']
model = LinearRegression()
model.fit(X, y)

Applications

Agricultural Planning: Forecast rice production to optimize planting schedules and resource allocation.
Climate Adaptation: Inform strategies to mitigate climate change impacts on agriculture.
Policy Development: Support government and stakeholders in ensuring food security in Sumatra.
Sustainability Analysis: Assess the ecological and economic sustainability of rice farming under climate change.

Acknowledgments

Built using Pandas, Matplotlib, Seaborn, and Scikit-learn.
Inspired by research on climate change impacts on rice production and machine learning forecasting methods.
Data sourced from Statistics Indonesia, Ministry of Agriculture, and climate studies.

License
This project is licensed under the MIT License. See the LICENSE file for details.
