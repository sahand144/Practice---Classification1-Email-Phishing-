# Email Phishing Detection Project

This project implements a machine learning model to detect phishing emails using a dataset of email features. The model uses Logistic Regression to classify emails as either legitimate or phishing attempts.

## Project Overview

The project performs the following steps:
1. Loads and preprocesses the email phishing dataset
2. Performs exploratory data analysis (EDA)
3. Splits the data into training and testing sets
4. Applies standardization to the features
5. Trains a Logistic Regression model
6. Evaluates the model's performance

## Dataset

The dataset is stored in a ZIP file containing a CSV file with various email features. The dataset includes:
- Multiple numeric features extracted from emails
- A binary label indicating whether an email is phishing (1) or legitimate (0)

## Requirements

The project requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Code Structure

The main script (`Email Phishing Dataset.py`) contains:
- Data loading and preprocessing
- Exploratory data analysis
- Model training and evaluation
- Visualization of results

## Results

The model's performance is evaluated using:
- Accuracy score
- Classification report
- Confusion matrix

## How to Run

1. Ensure you have all required packages installed
2. Place the dataset ZIP file in the correct location
3. Run the Python script:
   ```bash
   python "Email Phishing Dataset.py"
   ```

## Visualizations

The script generates several visualizations:
- Histograms of feature distributions
- Correlation heatmap
- Confusion matrix heatmap

## Future Improvements

Potential improvements for the project:
- Try different machine learning models
- Perform feature selection
- Implement cross-validation
- Add hyperparameter tuning
- Create a web interface for predictions
