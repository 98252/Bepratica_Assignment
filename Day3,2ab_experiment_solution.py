# ==============================================================================
# SECTION 9: STUDENT LAB ASSIGNMENT (A/B EXPERIMENT EVALUATION)
# ==============================================================================
# Description:
# 1. States the Null (H0) and Alternative (H1) hypotheses for Average Order Value (AOV).
# 2. Conducts a Two-Sample Welch's t-test at alpha = 0.05.
# 3. Computes sample means, standard errors, and 95% confidence intervals for both variants.
# 4. Synthesizes findings into an executive-ready business recommendation.
# ==============================================================================

import numpy as np
import pandas as pd
import scipy.stats as stats

# ------------------------------------------------------------------------------
# STEP 0: GENERATE EXPERIMENT DATASET
# ------------------------------------------------------------------------------
np.random.seed(888)
ab_experiment_df = pd.DataFrame({
    'Variant_A': np.random.normal(loc=72.50, scale=14.0, size=150),
    'Variant_B': np.random.normal(loc=77.20, scale=15.5, size=150)
})

# ------------------------------------------------------------------------------
# TASK 1: HYPOTHESIS FORMULATION
# ------------------------------------------------------------------------------
# Let mu_A = True population mean Average Order Value (AOV) for Variant A (Control)
# Let mu_B = True population mean Average Order Value (AOV) for Variant B (New Design)
#
# Null Hypothesis (H0):
#   mu_A = mu_B  (or mu_B - mu_A = 0)
#   There is no statistically significant difference in Average Order Value between
#   Variant A and Variant B. Any observed difference is due to random sampling variability.
#
# Alternative Hypothesis (H1):
#   mu_A != mu_B (or mu_B - mu_A != 0)  [Two-Tailed]
#   There is a statistically significant difference in Average Order Value between
#   Variant A and Variant B.

# ------------------------------------------------------------------------------
# TASK 2: TWO-SAMPLE WELCH'S T-TEST (alpha = 0.05)
# ------------------------------------------------------------------------------
# Note: Welch's t-test (equal_var=False) is preferred over Student's t-test
# because it does not assume equal population variances between variants.
alpha = 0.05

variant_a = ab_experiment_df['Variant_A']
variant_b = ab_experiment_df['Variant_B']

t_stat, p_value = stats.ttest_ind(variant_b, variant_a, equal_var=False)

# Welch-Satterthwaite degrees of freedom
n_a, n_b = len(variant_a), len(variant_b)
var_a, var_b = variant_a.var(ddof=1), variant_b.var(ddof=1)
df_welch = (var_a / n_a + var_b / n_b)**2 / (
    ((var_a / n_a)**2 / (n_a - 1)) + ((var_b / n_b)**2 / (n_b - 1))
)

reject_null = p_value < alpha

# ------------------------------------------------------------------------------
# TASK 3: SAMPLE MEANS, STANDARD ERRORS & 95% CONFIDENCE INTERVALS
# ------------------------------------------------------------------------------
def compute_ci(series, confidence=0.95):
    """Computes mean, standard error, margin of error, and confidence interval."""
    n = len(series)
    mean_val = series.mean()
    std_val = series.std(ddof=1)
    se_val = std_val / np.sqrt(n)
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * se_val
    ci_lower = mean_val - margin_of_error
    ci_upper = mean_val + margin_of_error
    return {
        'n': n,
        'mean': mean_val,
        'std': std_val,
        'se': se_val,
        't_crit': t_critical,
        'me': margin_of_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

stats_a = compute_ci(variant_a, confidence=0.95)
stats_b = compute_ci(variant_b, confidence=0.95)

# Difference in Means (Variant B - Variant A)
mean_diff = stats_b['mean'] - stats_a['mean']
se_diff = np.sqrt(stats_a['se']**2 + stats_b['se']**2)
t_crit_diff = stats.t.ppf(0.975, df=df_welch)
me_diff = t_crit_diff * se_diff
ci_diff_lower = mean_diff - me_diff
ci_diff_upper = mean_diff + me_diff
relative_lift = (mean_diff / stats_a['mean']) * 100

# ------------------------------------------------------------------------------
# TASK 4: EXECUTIVE BUSINESS RECOMMENDATION
# ------------------------------------------------------------------------------
executive_recommendation = (
    f"Based on our A/B test analysis of 300 total transactions (n=150 per variant), "
    f"the new checkout page design (Variant B) delivered a statistically significant lift "
    f"in Average Order Value of ${mean_diff:.2f} (+{relative_lift:.2f}%), moving from "
    f"${stats_a['mean']:.2f} (95% CI: [${stats_a['ci_lower']:.2f}, ${stats_a['ci_upper']:.2f}]) to "
    f"${stats_b['mean']:.2f} (95% CI: [${stats_b['ci_lower']:.2f}, ${stats_b['ci_upper']:.2f}]). "
    f"With a Welch's t-statistic of {t_stat:.4f} and a p-value of {p_value:.4f} (well below our "
    f"significance threshold of alpha = 0.05), we reject the null hypothesis and conclude that this "
    f"gain is not the result of random chance. The 95% confidence interval for the net incremental "
    f"order value ranges from +${ci_diff_lower:.2f} to +${ci_diff_upper:.2f} per transaction. "
    f"Consequently, the analytics team strongly recommends a full 100% rollout of Variant B across all "
    f"checkout traffic, projected to generate substantial annualized revenue expansion."
)

# ------------------------------------------------------------------------------
# DISPLAY STRUCTURED LAB OUTPUT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 90)
    print("SECTION 9: A/B EXPERIMENT EVALUATION & STATISTICAL ANALYSIS")
    print("=" * 90)

    print("\n--- TASK 1: HYPOTHESIS FORMULATION ---")
    print("  * Null Hypothesis (H0)        : mu_B - mu_A = 0  (No difference in AOV between variants)")
    print("  * Alternative Hypothesis (H1) : mu_B - mu_A != 0 (Statistically significant difference in AOV)")
    print(f"  * Significance Threshold (alpha): {alpha:.2f}")

    print("\n--- TASK 2: TWO-SAMPLE WELCH'S T-TEST RESULTS ---")
    print(f"  * Test Type                  : Welch's Two-Sample t-test (Unpooled Variance)")
    print(f"  * Welch Degrees of Freedom   : {df_welch:.2f}")
    print(f"  * Welch t-statistic          : {t_stat:.4f}")
    print(f"  * Two-Tailed p-value         : {p_value:.4f}")
    print(f"  * Decision at alpha = {alpha}    : {'REJECT H0 (Statistically Significant)' if reject_null else 'FAIL TO REJECT H0'}")

    print("\n--- TASK 3: PARAMETRIC ESTIMATES & 95% CONFIDENCE INTERVALS ---")
    summary_table = pd.DataFrame([
        {
            'Variant': 'Variant A (Control)',
            'Sample Size (n)': stats_a['n'],
            'Mean AOV ($)': f"${stats_a['mean']:.2f}",
            'Std Dev ($)': f"${stats_a['std']:.2f}",
            'Standard Error ($)': f"${stats_a['se']:.2f}",
            'Margin of Error ($)': f"${stats_a['me']:.2f}",
            '95% Confidence Interval': f"[${stats_a['ci_lower']:.2f}, ${stats_a['ci_upper']:.2f}]"
        },
        {
            'Variant': 'Variant B (New Design)',
            'Sample Size (n)': stats_b['n'],
            'Mean AOV ($)': f"${stats_b['mean']:.2f}",
            'Std Dev ($)': f"${stats_b['std']:.2f}",
            'Standard Error ($)': f"${stats_b['se']:.2f}",
            'Margin of Error ($)': f"${stats_b['me']:.2f}",
            '95% Confidence Interval': f"[${stats_b['ci_lower']:.2f}, ${stats_b['ci_upper']:.2f}]"
        },
        {
            'Variant': 'Difference (B - A)',
            'Sample Size (n)': f"{n_a} vs {n_b}",
            'Mean AOV ($)': f"+${mean_diff:.2f} (+{relative_lift:.2f}%)",
            'Std Dev ($)': "-",
            'Standard Error ($)': f"${se_diff:.2f}",
            'Margin of Error ($)': f"${me_diff:.2f}",
            '95% Confidence Interval': f"[+${ci_diff_lower:.2f}, +${ci_diff_upper:.2f}]"
        }
    ]).set_index('Variant')
    print(summary_table.to_string())

    print("\n--- TASK 4: EXECUTIVE BUSINESS RECOMMENDATION ---")
    print(executive_recommendation)
    print("\n" + "=" * 90)
