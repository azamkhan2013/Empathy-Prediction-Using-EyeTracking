#Empathy Prediction Project
This repository contains code and data for predicting empathy scores using eye-tracking data.

Getting Started
Prerequisites
Python 3.7 or later
Required packages: pandas, numpy, scikit-learn
Downloading the Data
The data can be downloaded from [https://www.nature.com/articles/s41597-022-01862-w#Sec10]. Please download and extract the data into the data folder in the root directory of the project.

Running the Code
To run the code, simply run the main.py.


python main.py
By default, the script will train and evaluate a Random Forest Regressor on the data using 10-fold cross-validation. The results will be printed to the console.

Repository Structure
main.py: the main script that runs the project.
preprocessing.py: contains functions for cleaning and preprocessing the data.
model.py: contains functions for training and evaluating the predictive model.
README.md: this file.
Author
Azam khan ([azamkhan2013@outlook.com])
