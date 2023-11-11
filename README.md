# SMILES-based Target Prediction App

> This app takes SMILES and Use rdkit descriptors to produce ML model

## Overview

This Streamlit web application is designed to predict target values from chemical SMILES data using machine learning models. SMILES (Simplified Molecular Input Line Entry System) is a notation that allows a user to represent a chemical structure in a way that can be used by the computer.

>The application provides the following features:

### Data Loading

Users can upload a CSV or Excel file containing SMILES and target columns.
Optionally, users can choose to load sample data.

### Data Exploration

The uploaded data is displayed in a tabular format for exploration. Users must specify the names of the SMILES and target columns.

### Feature Engineering

The application converts SMILES data into a feature matrix using RDKit. The feature matrix is displayed for further exploration.

### Model Selection

Users can choose a Randomforest machine learning model for training but more can be added. Currently, only the Random Forest model is available. Hyperparameter options are provided, including the ability to customize or use default values.

### Training the Model

Users can initiate model training with the specified hyperparameters. Model performance metrics, such as R-squared, Mean Absolute Error, and Mean Squared Error, are displayed.
Feature importances are visualized through a bar chart, highlighting the top 10 important features.

### Prediction

Users can enter a SMILES string to predict the target value using the trained model. The predicted value is displayed, and users have the option to export the trained model.

## Usage Instructions

### Upload Data

Use the file uploader to upload a CSV or Excel file containing SMILES and target columns. Optionally, check the "Load sample data" checkbox to load sample data.

### Explore Data

View the uploaded data in the displayed table.
Enter the names of the SMILES and target columns in the provided text inputs.

### Train Model

Select the "Train your model now?" section.
Choose the Random Forest model and specify hyperparameters (or use default values). Click "Start Training" to initiate model training. View the best hyperparameters, model performance metrics, and feature importances.

### Predict Target

In the "Do you want to Predict target?" section, enter a SMILES string. View the featurized SMILES data, an image of the molecule, and the predicted target value. Optionally, export the trained model by checking the "Happy to Export model?" checkbox.

## Requirements
> Python 3.7 or higher, Streamlit, scikit-learn,pandas, RDKit, matplotlib, numpy, joblib

## Installation

`git clone https://github.com/AW1176/`

`pip install -r requirements.txt`

`streamlit run app.py`

