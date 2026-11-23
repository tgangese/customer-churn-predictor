import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nChurn counts:")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True))
print("\nMissing:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\nDtypes:")
print(df.dtypes)
print("\nTotalCharges sample:")
print(df['TotalCharges'].head(10).tolist())

print("\n--- Cleaning ---")

# 1. Drop useless ID column
df = df.drop(columns=['customerID'])
print("Dropped customerID, new shape:", df.shape)

# 2. Fix TotalCharges: blank " " -> NaN, then to number
# Check how many blanks
print("TotalCharges blank spaces:", (df['TotalCharges'] == " ").sum())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("TotalCharges dtype after fix:", df['TotalCharges'].dtype)
print("Missing TotalCharges after fix:", df['TotalCharges'].isnull().sum())

# 3. Drop those few rows with missing TotalCharges (only 11 rows)
df = df.dropna()
print("Shape after dropna:", df.shape)

# 4. Check Churn target - convert Yes/No to 1/0 for ML
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\nChurn after mapping:")
print(df['Churn'].value_counts())

print("\n--- Encoding + Split + First Model ---")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Separate X and y
X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. One-Hot Encode all text columns (like you did for Product in sales project)
X = pd.get_dummies(X, drop_first=True)
print("X shape after encoding:", X.shape)
print("Columns:", X.columns.tolist()[:5], "...")

# 3. Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print("Train churn rate:", y_train.mean())
print("Test churn rate:", y_test.mean())

# 4. Train first model - Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Scaling + Compare Models ---")
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Scale X_train and X_test (fit only on train!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Logistic Regression WITH scaling
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Logistic Regression (scaled) - Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}, Recall Churn: {classification_report(y_test, y_pred_lr, output_dict=True)['1']['recall']:.3f}, F1 Churn: {f1_score(y_test, y_pred_lr):.3f}")

# 2. Random Forest (no scaling needed, trees don't care)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}, Recall Churn: {classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall']:.3f}, F1 Churn: {f1_score(y_test, y_pred_rf):.3f}")

# 3. Which is better for business?
print("\nConfusion Matrix Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n--- Improve Recall + Feature Importance ---")
from sklearn.linear_model import LogisticRegression as LR2

# Try class_weight='balanced' - penalize missing churners more
lr_bal = LR2(max_iter=1000, class_weight='balanced')
lr_bal.fit(X_train_scaled, y_train)
y_pred_bal = lr_bal.predict(X_test_scaled)
print(f"Logistic Balanced - Accuracy: {accuracy_score(y_test, y_pred_bal):.3f}, Recall Churn: {classification_report(y_test, y_pred_bal, output_dict=True)['1']['recall']:.3f}, F1: {f1_score(y_test, y_pred_bal):.3f}")
print("Confusion Balanced:")
print(confusion_matrix(y_test, y_pred_bal))

# Feature Importance - which features matter most?
# Use Random Forest for importance (even if accuracy lower, it tells us what matters)
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
print("\nTop 5 Features:")
print(feat_imp.head(5).to_string(index=False))
print("\nBottom 3 Features:")
print(feat_imp.tail(3).to_string(index=False))

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Re-train balanced model on FULL data for deployment (we already tested it)
# For simplicity, we re-run the pipeline quickly
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

# Save
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Saved: churn_model.pkl, scaler.pkl, columns.pkl")
print("Columns:", len(X.columns))