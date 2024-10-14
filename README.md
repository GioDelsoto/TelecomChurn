# Telecom Customer Churn Analysis

This repository contains a project focused on analyzing and predicting customer churn in the telecommunications industry. The analysis was performed using a dataset from Kaggle ([Telecom Customer Churn](https://www.kaggle.com/datasets/abhinav89/telecom-customer)). The project involves data exploration, feature engineering, model building, and evaluation, as well as a simple local server application for deploying the model.

## Project Structure

The repository is organized into the following folders:

- **app/**: Contains the application code, including `__init__.py`, configuration files, and route definitions for running the model on a local server.
  
- **data/**: Includes the project dataset, with both raw and processed data files. Data cleaning and preprocessing steps are applied to the raw data in this folder.

- **model/**: Contains the code for creating and managing the XGBoost model, including scripts for feature engineering, training, and evaluation.

- **notebook/**: Contains Jupyter notebooks with the main analyses, divided into three parts:
  1. **Data Exploration and Cleaning**: Initial data exploration and cleaning, such as handling missing values, outliers, and variable distributions.
  2. **Business-Driven Data Analysis**: Deeper data analysis to answer business-related questions and explore correlations between variables.
  3. **Modeling and Evaluation**: Comparison of machine learning models, hyperparameter tuning, and a profit simulation to assess the business impact of the model.

## Notebooks Overview

1. **Data Exploration and Cleaning**: Examines variable distributions and performs data cleaning, including handling missing values, duplicates, and outliers.

2. **Business-Driven Data Analysis**: Addresses specific business questions, explores correlations, and includes other exploratory analyses to guide the model-building process.

3. **Modeling and Evaluation**: Builds and compares different machine learning models, with a focus on XGBoost. Hyperparameter tuning is performed, and a profit simulation is included to evaluate the model's potential impact on business decisions.

## Local Server Application

A simple local server application is included to deploy the model, allowing for easy integration into a business environment where predictions can be made on new data.

---

Feel free to explore the folders and notebooks for a comprehensive view of the analysis and model development process. Contributions and suggestions are welcome!
