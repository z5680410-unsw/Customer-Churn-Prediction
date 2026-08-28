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
- Split data into 80% training / 20% testing sets (fixed `random_state=42` for reproducibility)
- Trained a Logistic Regression model to classify churn (Yes/No)
- Evaluated performance using Accuracy, Precision, Recall, and F1 Score
- Compared results against a majority-class baseline
- Interpreted feature coefficients to identify patterns, despite the model's weak fit

## Results
- **Accuracy: 0.478** — below the ~51.5% baseline of simply predicting the majority class (the dataset is close to balanced: ~51.5% churned vs. 48.5% not churned). The model performs worse than a naive guess.
- **Precision: 0.473, Recall: 0.474, F1 Score: 0.472** — all close to 0.5, consistent with a model that isn't picking up any real signal rather than one with a genuine but imperfect fit.
- Together, these metrics indicate the available features have essentially no predictive power for churn in this dataset, rather than the model narrowly underperforming.

## Key Insight
No feature shows a reliable relationship with churn. This mirrors a related finding from a companion project on the same dataset (predicting `Total_Spend` via linear regression), where individual features also showed near-zero correlation with the target. This is consistent with the dataset being generated for practice purposes, where the outcome variables weren't deliberately built to depend on the other columns. Given the model's performance falls below a naive baseline, the feature coefficients are treated as exploratory rather than reliable drivers of churn.

## Next Steps
Since the limitation appears to be signal in the dataset rather than the choice of a linear model, a natural next step would be testing non-linear models (e.g. Random Forest, Gradient Boosting) to check whether they can extract any pattern a linear decision boundary can't.

## Tools & Libraries
Python, pandas, scikit-learn, matplotlib

## Notebook
Open `Customer_Churn_Prediction.ipynb` directly on GitHub to view the full code, outputs, and visualisations.
