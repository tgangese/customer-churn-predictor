import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load same data as churn project
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Only customers who STAYED. We want to understand loyal customers, not churners.
stayed = df[df['Churn'] == 'No'].copy()
print(f"Stayed customers: {len(stayed)} / {len(df)}")

# Fix TotalCharges same trap as before - blank spaces
stayed['TotalCharges'] = pd.to_numeric(stayed['TotalCharges'], errors='coerce')
stayed = stayed.dropna()
print(f"After cleaning: {len(stayed)}")

# Pick features that describe VALUE + LOYALTY, not everything
# tenure = how long, MonthlyCharges = how much per month, TotalCharges = total spent
X = stayed[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
print("\nFeatures for clustering:")
print(X.describe().round(1))

# K-Means uses distance, so scale. Same rule as Logistic.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Find best K with Elbow ---")
# Inertia = how far customers are from their center. Lower = tighter groups.
# As K grows, inertia always drops. We look for elbow where dropping slows.
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"K={k}, Inertia={km.inertia_:.0f}")

# Plot elbow
plt.figure(figsize=(7, 4))
plt.plot(K_range, inertias, marker='o')
plt.xlabel("K - number of segments")
plt.ylabel("Inertia - lower is tighter")
plt.title("Elbow Method - Where does it bend?")
plt.grid(True, alpha=0.3)
plt.savefig("elbow.png")
print("\nSaved elbow.png - open it and look for the bend")

print("\n--- Train final model with K=4 ---")
# From elbow, K=3 is usually the bend for this data. Check your plot.
# If your elbow looks like K=4, change this to 4.
k_final = 4
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
stayed['Segment'] = kmeans.fit_predict(X_scaled)

print(f"\nCustomers per segment:")
print(stayed['Segment'].value_counts().sort_index())

print("\n--- What does each segment look like? ---")
# Group by segment and average - this is how you NAME segments
summary = stayed.groupby('Segment')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean().round(1)
print(summary)

# Also check Contract + Internet to understand behavior
print("\nContract mix per segment:")
print(pd.crosstab(stayed['Segment'], stayed['Contract'], normalize='index').round(2))

print("\nInternet mix per segment:")
print(pd.crosstab(stayed['Segment'], stayed['InternetService'], normalize='index').round(2))

print("\n--- Save for later ---")
stayed[['tenure', 'MonthlyCharges', 'TotalCharges', 'Segment']].to_csv("segments.csv", index=False)
print("Saved segments.csv + elbow.png")