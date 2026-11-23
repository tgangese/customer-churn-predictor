# Customer Churn Predictor

Predict which telecom customers will leave, so company can offer retention deal before they leave.

## Results
- Dataset: 7,032 customers, 21 columns (Telco Customer Churn)
- Best Model: Logistic Regression (balanced, scaled)
- Accuracy: 72.6%, Recall (Churn): 79.7%, F1: 60.8%
- Why Recall matters: Missing churner costs $500, false alarm costs $10

## Top Features
1. TotalCharges, 2. tenure, 3. MonthlyCharges

## How to Run
pip install pandas scikit-learn
python3 churn.py