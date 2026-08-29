# Customer Churn Prediction (Logistic Regression + Model Comparison)

Predicting whether an online retail customer will churn, using logistic regression on behavioural and demographic data, then comparing performance against three additional classifiers.

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
- Trained three additional classifiers — Random Forest, Gradient Boosting, and SVM — on the same train/test split, to test whether non-linear models or a different approach could extract signal a linear model couldn't
- Compared all four models side by side, including a bar chart against the baseline

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.478 | 0.473 | 0.474 | 0.472 |
| Random Forest | 0.478 | 0.479 | 0.479 | 0.478 |
| Gradient Boosting | 0.500 | 0.500 | 0.500 | 0.500 |
| SVM | 0.471 | 0.465 | 0.466 | 0.463 |

- The majority-class baseline is ~0.515 (the dataset is close to balanced: ~51.5% churned vs. 48.5% not churned). **All four models fall at or below this baseline** — none of them meaningfully outperform a naive guess.
- Gradient Boosting comes closest to the baseline (0.500); Logistic Regression and SVM fall below it.
- Precision, Recall, and F1 Score cluster around 0.46–0.50 for every model, consistent with models that aren't picking up any real signal, rather than models with a genuine but modest fit.

## Key Insight
No model — linear or non-linear — was able to meaningfully beat the baseline, despite testing four quite different approaches (a linear classifier, two tree-based ensembles, and a kernel-based method). This is a stronger conclusion than testing Logistic Regression alone: it indicates the limitation lies in the dataset itself rather than the choice of algorithm. This mirrors a related finding from a companion project on the same dataset (predicting `Total_Spend` via linear regression), where individual features also showed near-zero correlation with the target — consistent with the dataset being generated for practice purposes, where the outcome variables weren't deliberately built to depend on the other columns.

## Next Steps
Since four different modelling approaches all hit the same ceiling, further model testing on these same features is unlikely to help. A more productive next step would be revisiting the available features and data collection process — for example, gathering additional behavioural data that might carry a stronger relationship with churn.

## Tools & Libraries
Python, pandas, scikit-learn, matplotlib

## Notebook
Open `Customer_Churn_Prediction.ipynb` directly on GitHub to view the full code, outputs, and visualisations.
