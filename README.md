# Eyes Don’t Lie: Uncovering the Power of Eye-Tracking to Evaluate Empathy in Recruitment

## Overview

This repository contains the code and findings from my project titled "Eyes Don’t Lie: Uncovering the Power of Eye-Tracking to Evaluate Empathy in Recruitment," completed on April 24, 2023. The study focuses on using eye-tracking data to predict empathy scores in recruitment contexts, utilizing machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Main Findings](#main-findings)
- [Methodology](#methodology)
- [Discussion and Results](#discussion-and-results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## Introduction

Empathy is crucial in human interactions, aiding in building social connections, improving communication, and deepening understanding. This study leverages the EyeT4 dataset to assess empathy during recruitment through eye-gaze movements. Using regression models, including RandomForestRegressor and GradientBoostingRegressor with GroupKFold cross-validation, we aimed to predict empathy scores accurately.

## Main Findings

The study analyzed data from 60 participants divided into a control group and a test group. The test group performed gaze typing tasks, requiring empathy, while the control group viewed structureless images. Empathy scores were obtained from a questionnaire dataset completed post-experiment.

### Exploratory Data Analysis

1. **Data Exploration**: The initial dataset contained 3.7 million rows and 71 columns with many null values and variables with no variance. Further data cleaning was necessary.
   
2. **Data Cleaning and Transformation**: Pupil dilation measurements significantly impacted the target variable but contained many null values. Columns with over 70% null values were removed. Missing values were addressed using simple and iterative imputation.

3. **Data Visualization and Feature Extraction**: Histograms, time series plots, and correlation heat maps were used to extract key features. Variables like participant name, recording timestamps, gaze points, and pupil diameters were identified as informative for predicting empathy scores.

## Methodology

### Building Model

We used RandomForestRegressor to predict empathy scores, evaluated using mean squared error (MSE) and r2 score metrics. Initial results showed a low MSE of 0.008 and a high r2 score of 0.99, indicating potential data leakage due to fixed empathy scores.

### Optimization

To improve results, we combined all data into a single file, scaled the dataset using RobustScaler, and incorporated GroupKFold validation to prevent data leakage. Additionally, GradientBoostingRegressor was used, achieving an r2 value of 0.11 and an MSE of 0.7.

## Discussion and Results

Our models demonstrated that the dataset was useful for predicting empathy scores, emphasizing the importance of data cleaning and feature extraction. However, limitations such as potential data leakage, small sample size, and reliance on eye-gaze movements alone were noted. Future research should explore additional predictors like facial expressions and physiological responses.

## Conclusion

The project successfully predicted empathy scores using eye-gaze movements. It highlighted the critical role of data preparation in developing accurate models. For practical applications, companies should ensure high-quality datasets and thorough data cleaning. Future enhancements could include integrating other predictors to improve model robustness.

## Usage

To run the code and reproduce the results:

1. Clone this repository:
   ```sh
   git clone https://github.com/azamkhan2013/Empathy_estimator.git

## Downloading the Data
The data can be downloaded from [https://www.nature.com/articles/s41597-022-01862-w#Sec10]. Please download and extract the data into the data folder in the root directory of the project.

## Running the Code
To run the code, simply run the main.py.


