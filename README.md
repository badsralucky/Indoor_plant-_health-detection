# Indoor_plant-_health-detection
Built a machine learning model to predict plant health using environmental data.
Optimized model reliability using Stratified 5-Fold Cross-Validation and feature-importance visualizations
This repository contains a machine learning project designed to predict the Health Score of indoor plants based on environmental factors, maintenance routines, and physical observations. The core of the project is a CatBoost Multiclass Classifier implemented in a Jupyter Notebook, optimized for handling both categorical and numerical sensor-like data.

📋 Project Overview
Indoor plant health is influenced by a complex interaction of variables such as sunlight exposure, watering frequency, soil moisture, and pest presence. This project utilizes a dataset of 2,000 plant records to:

Perform Exploratory Data Analysis (EDA) to understand data distribution and missingness.

Engineer features like Water_per_watering, Temp_Humidity_interaction, and Sunlight_hours.

Train a robust CatBoost model using Stratified K-Fold Cross-Validation.

Evaluate performance using Macro-F1 Score and Confusion Matrices to account for class imbalance.
