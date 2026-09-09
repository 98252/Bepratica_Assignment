# ==============================================================================
# SECTION 9: STUDENT PRACTICE LAB — STATISTICAL DIAGNOSTICS & SKEW PROFILING
# ==============================================================================
# Description:
# 1. Computes Mean, Median, Std Dev, IQR, Skewness, and Count of Outliers (1.5x IQR).
# 2. Diagnoses the skewness direction (Left-Skewed, Symmetric, or Right-Skewed).
# 3. Determines the optimal measure of central tendency (Mean vs Median) for business decisions.
# ==============================================================================

import numpy as np
import pandas as pd
import scipy.stats as stats

# ------------------------------------------------------------------------------
# STEP 0: GENERATE STUDENT LAB DATASET
# ------------------------------------------------------------------------------
np.random.seed(999)
market_df = pd.DataFrame({
    'Annual_Income': np.random.exponential(scale=35_000, size=500) + 15_000,
    'Test_Score': 100 - np.random.exponential(scale=10, size=500).clip(0, 45),
    'Product_Weight': np.random.normal(loc=5.0, scale=0.5, size=500)
})

# ------------------------------------------------------------------------------
# TASKS 1, 2, AND 3 IMPLEMENTATION
# ------------------------------------------------------------------------------
results = []

for col in market_df.columns:
    series = market_df[col]
    
    # Task 1: Parametric and Non-Parametric Metrics
    mean_val = series.mean()
    median_val = series.median()
    std_val = series.std(ddof=1)  # Bessel-corrected sample standard deviation
    
    # Quartiles & 1.5x IQR Tukey Fences for Outlier Detection
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_val = q3 - q1
    lower_fence = q1 - 1.5 * iqr_val
    upper_fence = q3 + 1.5 * iqr_val
    
    outliers = series[(series < lower_fence) | (series > upper_fence)]
    outlier_count = len(outliers)
    
    # Skewness
    skew_val = series.skew()
    
    # Task 2: Diagnose Skewness Direction
    # Criteria:
    # Skewness > +0.5        -> Right-Skewed (Positive Skew, Mean > Median)
    # Skewness < -0.5        -> Left-Skewed (Negative Skew, Mean < Median)
    # -0.5 <= Skewness <= 0.5 -> Symmetric (Bell-shaped, Mean ≈ Median)
    if skew_val > 0.5:
        skew_diag = "Right-Skewed (Positive Skew)"
    elif skew_val < -0.5:
        skew_diag = "Left-Skewed (Negative Skew)"
    else:
        skew_diag = "Symmetric (Bell-shaped)"
        
    # Task 3: Central Tendency Recommendation for Business Decisions
    # When distribution is heavily skewed (|skew| > 0.5), use Median.
    # When distribution is symmetric (|skew| <= 0.5), use Mean.
    if abs(skew_val) > 0.5:
        recommended_metric = "Median"
        business_rationale = (
            f"Heavily skewed distribution ({skew_diag}). The Mean is pulled toward the long tail "
            f"(Mean = {mean_val:,.2f} vs Median = {median_val:,.2f}). The Median is robust to outliers "
            f"and best represents the typical customer/observation."
        )
    else:
        recommended_metric = "Mean"
        business_rationale = (
            f"Symmetric distribution (Skewness = {skew_val:.2f} ~= 0, Mean ~= Median). The Mean is the "
            f"most efficient unbiased estimator and utilizes information from all sample points."
        )
        
    results.append({
        'Variable': col,
        'Mean': round(mean_val, 2),
        'Median': round(median_val, 2),
        'Std Dev': round(std_val, 2),
        'IQR': round(iqr_val, 2),
        'Skewness': round(skew_val, 2),
        'Outlier Count': outlier_count,
        'Skewness Diagnosis': skew_diag,
        'Recommended Metric': recommended_metric,
        'Business Rationale': business_rationale
    })

diagnostics_summary = pd.DataFrame(results).set_index('Variable')

# ------------------------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 105)
    print("SECTION 9: STUDENT PRACTICE LAB - STATISTICAL DIAGNOSTICS & SKEW PROFILING")
    print("=" * 105)
    print("\n--- TASK 1 & 2: SUMMARY STATISTICS, SKEWNESS & OUTLIERS ---")
    print(diagnostics_summary[[
        'Mean', 'Median', 'Std Dev', 'IQR', 'Skewness', 'Outlier Count', 'Skewness Diagnosis'
    ]].to_string())

    print("\n" + "=" * 105)
    print("--- TASK 3: CENTRAL TENDENCY RECOMMENDATIONS & BUSINESS DECISION RATIONALE ---")
    print("=" * 105)
    for row in results:
        print(f"\n[Variable: {row['Variable']}]")
        print(f"  * Skewness Profile   : {row['Skewness Diagnosis']} (Skewness = {row['Skewness']:+.2f})")
        print(f"  * Central Tendency   : Mean = {row['Mean']:,} | Median = {row['Median']:,}")
        print(f"  * Outliers Flagged   : {row['Outlier Count']} outliers (via 1.5x IQR Tukey Rule)")
        print(f"  * Recommended Metric : {row['Recommended Metric']}")
        print(f"  * Business Rationale : {row['Business Rationale']}")
    print("\n" + "=" * 105)
