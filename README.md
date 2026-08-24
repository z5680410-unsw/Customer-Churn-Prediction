# Customer Churn Prediction (Logistic Regression)

Predicting whether an online retail customer will churn, using logistic regression on behavioural and demographic data.

## Problem
Losing customers is costly — acquiring a new one is far more expensive than retaining an existing one. Identifying customers at risk of churning early lets a business intervene with targeted retention efforts before they leave.

## Data
Online retail customer dataset including attributes such as age, gender, annual income, total spend, years as customer, number of purchases, number of returns, support contacts, satisfaction score, email opt-in status, and promotion response, with `Churned` as the target variable.

## Approach
- Checked and handled missing values
- Converted categorical variables (Gender, Promotion_Response) into dummy/indicator variables
- Scaled numeric features using StandardScaler
- Split data into 80% training / 20% testing sets
- Trained a Logistic Regression model to classify churn (Yes/No)
- Evaluated performance using Accuracy, Precision, Recall, and F1 Score
- Interpreted feature coefficients to identify key churn drivers

## Results
- **Accuracy:** ~47% — comparable to a coin flip, suggesting the available features have limited predictive power for churn on their own
- **Precision / Recall / F1:** used to assess how well the model distinguishes churners from non-churners, since accuracy alone can be misleading

## Key Insight
Purchase behaviour and gender emerged as the strongest predictors of churn, while satisfaction score had a comparatively weak effect — suggesting churn is driven more by engagement patterns than reported satisfaction.

## Tools & Libraries
Python, pandas, scikit-learn, matplotlib

## Notebook
Open the `.ipynb` file directly on GitHub to view the full code, outputs, and visualisations.
