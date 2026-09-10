# ==============================================================================
# SECTION 8: STUDENT GRADED LAB ASSIGNMENT — BEHAVIORAL CUSTOMER SEGMENTATION
# ==============================================================================


import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Safe display helper for both Jupyter/Colab and standalone Python environments
try:
    from IPython.display import display
except ImportError:
    display = print

# ------------------------------------------------------------------------------
# STEP 0: DATA INGESTION (customers.csv)
# ------------------------------------------------------------------------------
DATA_FILE = "customers.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"⚠️ '{DATA_FILE}' not found! Please ensure 'customers.csv' is placed in your working directory."
    )

df_customers = pd.read_csv(DATA_FILE)
print(f"✅ Successfully loaded '{DATA_FILE}' from disk.")
print(f"Dataset Dimensions: {df_customers.shape[0]} customers × {df_customers.shape[1]} features\n")
print("First 3 Records:")
display(df_customers.head(3))

# ------------------------------------------------------------------------------
# 1. Feature selection and scaling
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 1: FEATURE SELECTION & STANDARD SCALING")
print("=" * 80)

behavioral_features = ["Total_Transactions", "Avg_Order_Value", "Web_Engagement_Score"]
X_behavioral_raw = df_customers[behavioral_features]

scaler = StandardScaler()
X_behavioral_scaled = scaler.fit_transform(X_behavioral_raw)

print(f"Selected Behavioral Features : {behavioral_features}")
print(f"Scaled Feature Matrix Shape  : {X_behavioral_scaled.shape}")
print(f"Mean after scaling (approx 0): {np.round(X_behavioral_scaled.mean(axis=0), 4)}")
print(f"Std after scaling  (unit var): {np.round(X_behavioral_scaled.std(axis=0), 4)}")

# ------------------------------------------------------------------------------
# 2. Fit K-Means (K=3, random_state=42)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 2: FIT K-MEANS CLUSTERING (K=3, RANDOM_STATE=42)")
print("=" * 80)

K = 3
kmeans_model = KMeans(
    n_clusters=K,
    init="k-means++",
    n_init=10,
    random_state=42
)
behavioral_labels = kmeans_model.fit_predict(X_behavioral_scaled)
print(f"K-Means model successfully fitted with K={K}, random_state=42.")
print(f"Cluster label counts: {np.bincount(behavioral_labels)}")

# ------------------------------------------------------------------------------
# 3. Compute Silhouette Score
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 3: SILHOUETTE SCORE EVALUATION")
print("=" * 80)

score = silhouette_score(X_behavioral_scaled, behavioral_labels)
print(f"Behavioral 3-Cluster Model Silhouette Score: {score:.4f}")

# ------------------------------------------------------------------------------
# 4. Attach and Profile
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 4: ATTACH LABELS & GENERATE STATISTICAL PROFILES")
print("=" * 80)

# Attach cluster assignments to customer dataframe
df_customers["Behavioral_Cluster"] = behavioral_labels

# Statistical Profile on the 3 behavioral features
profile_behavioral = (
    df_customers
    .groupby("Behavioral_Cluster")[behavioral_features]
    .mean()
    .round(2)
)
profile_behavioral["Customer_Count"] = df_customers["Behavioral_Cluster"].value_counts()
profile_behavioral["Population_Share_%"] = (
    df_customers["Behavioral_Cluster"].value_counts(normalize=True) * 100
).round(1)

# Sort by cluster index
profile_behavioral = profile_behavioral.sort_index()

print("\n=== BEHAVIORAL CLUSTER STATISTICAL PROFILE TABLE (K=3) ===")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
display(profile_behavioral)

# Full context profile across all features
full_profile = (
    df_customers
    .groupby("Behavioral_Cluster")[["Age", "Annual_Income_kUSD", "Spending_Score"] + behavioral_features]
    .mean()
    .round(2)
)
full_profile["Customer_Count"] = df_customers["Behavioral_Cluster"].value_counts()
full_profile["Population_Share_%"] = (
    df_customers["Behavioral_Cluster"].value_counts(normalize=True) * 100
).round(1)
full_profile = full_profile.sort_index()

print("\n=== COMPREHENSIVE CLUSTER PROFILE (ALL ATTRIBUTES) ===")
display(full_profile)

# ------------------------------------------------------------------------------
# 5. Assign business persona names based on cluster centroids
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TASK 5: BUSINESS PERSONA MAPPING & STRATEGIC RECOMMENDATIONS")
print("=" * 80)

# Persona dictionary mapping cluster ID to commercial profiles
cluster_personas = {
    0: {
        "Business_Title": "Low-Touch / Occasional Buyers",
        "Statistical_Signature": f"Lowest Transactions ({profile_behavioral.loc[0, 'Total_Transactions']:.1f}), Modest AOV (${profile_behavioral.loc[0, 'Avg_Order_Value']:.2f}), Low Web Engagement ({profile_behavioral.loc[0, 'Web_Engagement_Score']:.1f})",
        "Demographic_Profile": f"Mature buyers (Avg Age: {full_profile.loc[0, 'Age']:.1f}), lowest spending score ({full_profile.loc[0, 'Spending_Score']:.1f}).",
        "Recommended_Action": "Reactivation win-back emails, essential bundles, free-shipping threshold incentives, deep seasonal clearance discounts."
    },
    1: {
        "Business_Title": "VIP Omnichannel Champions",
        "Statistical_Signature": f"Highest Transactions ({profile_behavioral.loc[1, 'Total_Transactions']:.1f}), Highest AOV (${profile_behavioral.loc[1, 'Avg_Order_Value']:.2f}), Highest Web Engagement ({profile_behavioral.loc[1, 'Web_Engagement_Score']:.1f})",
        "Demographic_Profile": f"Affluent young professionals (Avg Age: {full_profile.loc[1, 'Age']:.1f}, Income: ${full_profile.loc[1, 'Annual_Income_kUSD']:.1f}k, Spending Score: {full_profile.loc[1, 'Spending_Score']:.1f}).",
        "Recommended_Action": "Exclusive VIP perks, white-glove concierge service, private early access to premium product drops, high-tier loyalty rewards."
    },
    2: {
        "Business_Title": "Digital Frequent Browsers",
        "Statistical_Signature": f"Moderate Transactions ({profile_behavioral.loc[2, 'Total_Transactions']:.1f}), Standard AOV (${profile_behavioral.loc[2, 'Avg_Order_Value']:.2f}), Strong Digital Engagement ({profile_behavioral.loc[2, 'Web_Engagement_Score']:.1f})",
        "Demographic_Profile": f"Core active customer base (Avg Age: {full_profile.loc[2, 'Age']:.1f}, Spending Score: {full_profile.loc[2, 'Spending_Score']:.1f}). Largest cohort ({profile_behavioral.loc[2, 'Population_Share_%']}%).",
        "Recommended_Action": "Targeted push notifications, cross-sell and upsell add-ons at checkout, loyalty points for daily app visits, flash promotions."
    }
}

print("=== BUSINESS PERSONA DICTIONARY ===")
for cid, pdata in cluster_personas.items():
    print(f"\n🏷️  Cluster {cid}: '{pdata['Business_Title']}'")
    print(f"    * Signature   : {pdata['Statistical_Signature']}")
    print(f"    * Demographic : {pdata['Demographic_Profile']}")
    print(f"    * Commercial  : {pdata['Recommended_Action']}")

print("\n" + "=" * 80)
print("✅ STUDENT GRADED LAB ASSIGNMENT COMPLETED SUCCESSFULLY!")
print("=" * 80)
