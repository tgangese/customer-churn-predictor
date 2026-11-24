import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score

# Load data
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

# customerID is unique for everyone, model would just memorise it. Drop it.
df = df.drop(columns=['customerID'])
print("Dropped customerID, new shape:", df.shape)

# TotalCharges looks like numbers but pandas reads it as text.
# Found 11 blank spaces " " hiding inside. That's why dtype is str.
print("TotalCharges blank spaces:", (df['TotalCharges'] == " ").sum())

# errors='coerce' turns anything that isn't a number into NaN instead of crashing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("TotalCharges dtype after fix:", df['TotalCharges'].dtype)
print("Missing TotalCharges after fix:", df['TotalCharges'].isnull().sum())

# Only 11 rows bad out of 7043, not worth filling. Just drop them.
df = df.dropna()
print("Shape after dropna:", df.shape)

# Yes/No -> 1/0 so model can learn it
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\nChurn after mapping:")
print(df['Churn'].value_counts())

print("\n--- Encoding + Split + First Model ---")

X = df.drop(columns=['Churn'])
y = df['Churn']

# Turn text columns like Contract, InternetService into 0/1 columns
X = pd.get_dummies(X, drop_first=True)
print("X shape after encoding:", X.shape)
print("Columns:", X.columns.tolist()[:5], "...")

# stratify keeps same churn % in train and test, otherwise test could be lucky/unlucky
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print("Train churn rate:", y_train.mean())
print("Test churn rate:", y_test.mean())

# First try, no scaling yet
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Scaling + Compare Models ---")

# Logistic needs scaling, trees don't. Fit scaler on train only, or we leak test info.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Same logistic but with scaled data - should stop the convergence warning
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
print(f"Logistic Regression (scaled) - Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}, Recall Churn: {classification_report(y_test, y_pred_lr, output_dict=True)['1']['recall']:.3f}, F1 Churn: {f1_score(y_test, y_pred_lr):.3f}")

# Random Forest for comparison. No scaling needed here.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}, Recall Churn: {classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall']:.3f}, F1 Churn: {f1_score(y_test, y_pred_rf):.3f}")

print("\nConfusion Matrix Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))

print("\n--- Improve Recall + Feature Importance ---")

# Normal model misses too many churners. Balanced tells model: missing a churner is worse.
lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_bal.fit(X_train_scaled, y_train)
y_pred_bal = lr_bal.predict(X_test_scaled)
print(f"Logistic Balanced - Accuracy: {accuracy_score(y_test, y_pred_bal):.3f}, Recall Churn: {classification_report(y_test, y_pred_bal, output_dict=True)['1']['recall']:.3f}, F1: {f1_score(y_test, y_pred_bal):.3f}")
print("Confusion Balanced:")
print(confusion_matrix(y_test, y_pred_bal))

# Which features actually matter? Using RF just for importance, even though logistic won.
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
print("\nTop 5 Features:")
print(feat_imp.head(5).to_string(index=False))
print("\nBottom 3 Features:")
print(feat_imp.tail(3).to_string(index=False))

print("\n--- Cross-Validation + Dollar Cost ---")
# One split can be lucky. CV checks 5 different splits.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_bal, X_train_scaled, y_train, cv=cv, scoring='recall')
print(f"CV Recall: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print("CV scores per fold:", [round(s, 3) for s in cv_scores])

# Turn metrics into money: missed churner = $500, false alarm = $10
# confusion = [[TN, FP], [FN, TP]]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bal).ravel()
cost_balanced = 500 * fn + 10 * fp
print(f"\nBalanced model cost: ${cost_balanced} (FN={fn} x $500 + FP={fp} x $10)")

# Compare: what would normal model cost?
tn2, fp2, fn2, tp2 = confusion_matrix(y_test, y_pred_lr).ravel()
cost_normal = 500 * fn2 + 10 * fp2
print(f"Normal model cost: ${cost_normal} (FN={fn2} x $500 + FP={fp2} x $10)")
print(f"Saved by balanced model: ${cost_normal - cost_balanced}")
print(f"Balanced Precision: {precision_score(y_test, y_pred_bal):.3f} - we trade some precision for recall")

print("\n--- Save model for later ---")
# Retrain balanced model on full data so we can use it without retraining every time
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)
final_model = LogisticRegression(max_iter=1000, class_weight='balanced')
final_model.fit(X_scaled_full, y)

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler_full, f)
with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Saved: churn_model.pkl, scaler.pkl, columns.pkl")
print("Columns:", len(X.columns))

print("\n--- How to use saved model on new data ---")
# New data might have a category we never saw, e.g. new Contract type.
# get_dummies would make different columns -> model breaks.
# Fix: reindex to saved columns, fill missing with 0.
# Example:
# X_new = pd.get_dummies(new_df, drop_first=True)
# X_new = X_new.reindex(columns=list(X.columns), fill_value=0)
# X_new_scaled = scaler_full.transform(X_new)
# pred = final_model.predict(X_new_scaled)
print("Rule: Always reindex new data to saved columns before predicting. See comments above.")