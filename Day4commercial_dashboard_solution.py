# ==============================================================================
# SECTION 6: STUDENT LAB ASSIGNMENT (PORTFOLIO DELIVERABLE)
# ==============================================================================
# Commercial Operations & Regional Health Dashboard (2x2 Panel)
#
# MANDATORY SPECIFICATIONS:
# 1. Grid Layout: plt.subplots(2, 2, figsize=(14, 9)) in Object-Oriented style.
# 2. Panel 1 (Trend): Temporal line plot showing monthly net revenue and rolling average.
# 3. Panel 2 (Distribution): Histogram with KDE of Profit Margin (%).
# 4. Panel 3 (Comparison): Regional revenue performance bar chart with data labels.
# 5. Panel 4 (Relationship): Discount Rate vs Profit Margin regression plot.
# 6. Professional Polishing: Currency & percent formatting, clean themes, annotations.
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ------------------------------------------------------------------------------
# STEP 0: ENVIRONMENT SETUP & SYNTHETIC DATASET GENERATION
# ------------------------------------------------------------------------------
np.random.seed(42)
n_samples = 1500

date_range = pd.date_range(start="2024-01-01", end="2024-12-31", periods=n_samples)
regions = np.random.choice(["North America", "Europe", "Asia-Pacific", "Latin America"], size=n_samples, p=[0.40, 0.30, 0.20, 0.10])
customer_tier = np.random.choice(["Standard", "Silver", "Gold", "Platinum"], size=n_samples, p=[0.50, 0.25, 0.15, 0.10])
product_category = np.random.choice(["Hardware", "SaaS Subscriptions", "Consulting", "Support Add-ons"], size=n_samples, p=[0.35, 0.30, 0.20, 0.15])

units = np.random.poisson(lam=4, size=n_samples).clip(1, 20)
base_prices = {"Hardware": 450.0, "SaaS Subscriptions": 120.0, "Consulting": 250.0, "Support Add-ons": 75.0}
unit_price = np.array([base_prices[cat] for cat in product_category]) * np.random.uniform(0.85, 1.15, size=n_samples)

discount_pct = np.random.beta(a=2, b=8, size=n_samples) * 0.40
gross_revenue = units * unit_price
net_revenue = gross_revenue * (1 - discount_pct)
cost_price = gross_revenue * np.random.uniform(0.40, 0.65, size=n_samples)
profit_margin = ((net_revenue - cost_price) / net_revenue) * 100
csat_score = (np.random.normal(loc=4.2, scale=0.6, size=n_samples) - (discount_pct * 1.5)).clip(1.0, 5.0)

df = pd.DataFrame({
    "Date": date_range,
    "Region": regions,
    "Customer_Tier": customer_tier,
    "Category": product_category,
    "Units_Sold": units,
    "Unit_Price": np.round(unit_price, 2),
    "Gross_Revenue": np.round(gross_revenue, 2),
    "Discount_Pct": np.round(discount_pct, 4),
    "Net_Revenue": np.round(net_revenue, 2),
    "Profit_Margin_Pct": np.round(profit_margin, 2),
    "CSAT_Score": np.round(csat_score, 2)
})

df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
df["Month_Name"] = df["Date"].dt.strftime("%b")

# Format Helpers
def currency_k_formatter(x, pos):
    return f"${x * 1e-3:.0f}K" if abs(x) >= 1000 else f"${x:.0f}"

# Global aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ------------------------------------------------------------------------------
# STEP 1: INITIALIZE 2x2 FIGURE CANVAS
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Commercial Operations & Regional Health Dashboard (FY2024)", fontsize=16, fontweight="bold", y=0.98)

# ------------------------------------------------------------------------------
# PANEL [0, 0]: TREND ANALYSIS (Monthly Net Revenue & Rolling Average)
# ------------------------------------------------------------------------------
monthly_agg = df.groupby("Month")["Net_Revenue"].sum().reset_index()
monthly_agg["Rolling_3M"] = monthly_agg["Net_Revenue"].rolling(window=3, min_periods=1).mean()

axes[0, 0].plot(
    monthly_agg["Month"], monthly_agg["Net_Revenue"],
    marker="o", markersize=6, color="#1e3a8a", linewidth=2.2, label="Monthly Revenue"
)
axes[0, 0].plot(
    monthly_agg["Month"], monthly_agg["Rolling_3M"],
    linestyle="--", color="#d97706", linewidth=1.8, label="3-Month Moving Avg"
)

axes[0, 0].set_title("1. Monthly Net Revenue Run-Rate Trend", fontsize=12, fontweight="bold", loc="left")
axes[0, 0].set_ylabel("Net Revenue ($)", fontweight="medium")
axes[0, 0].yaxis.set_major_formatter(ticker.FuncFormatter(currency_k_formatter))
axes[0, 0].tick_params(axis='x', rotation=30)
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend(frameon=True, facecolor="white", loc="lower right")

# Peak Revenue Annotation
peak_rev = monthly_agg["Net_Revenue"].max()
peak_date = monthly_agg.loc[monthly_agg["Net_Revenue"].idxmax(), "Month"]
axes[0, 0].annotate(
    f"Peak: ${peak_rev * 1e-3:.1f}K",
    xy=(peak_date, peak_rev),
    xytext=(peak_date, peak_rev + (peak_rev * 0.08)),
    arrowprops=dict(facecolor="#dc2626", shrink=0.08, width=1.2, headwidth=6),
    fontweight="bold", color="#dc2626", fontsize=9.5, ha='center'
)

# ------------------------------------------------------------------------------
# PANEL [0, 1]: DISTRIBUTION ANALYSIS (Profit Margin Spread & Median)
# ------------------------------------------------------------------------------
sns.histplot(
    df["Profit_Margin_Pct"], kde=True, ax=axes[0, 1],
    color="#0284c7", edgecolor="white", alpha=0.75, bins=25
)

median_margin = df["Profit_Margin_Pct"].median()
axes[0, 1].axvline(
    median_margin, color="#dc2626", linestyle="--", linewidth=2,
    label=f"Median Margin: {median_margin:.1f}%"
)

axes[0, 1].set_title("2. Profit Margin Distribution", fontsize=12, fontweight="bold", loc="left")
axes[0, 1].set_xlabel("Profit Margin (%)", fontweight="medium")
axes[0, 1].set_ylabel("Transaction Count", fontweight="medium")
axes[0, 1].xaxis.set_major_formatter(ticker.PercentFormatter())
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend(frameon=True, facecolor="white", loc="upper left")

# ------------------------------------------------------------------------------
# PANEL [1, 0]: CATEGORICAL COMPARISON (Total Revenue by Sales Region)
# ------------------------------------------------------------------------------
region_rev = df.groupby("Region")["Net_Revenue"].sum().sort_values(ascending=False).reset_index()

bar_colors = ["#1e40af", "#2563eb", "#60a5fa", "#93c5fd"]
bars = axes[1, 0].bar(
    region_rev["Region"], region_rev["Net_Revenue"],
    color=bar_colors, width=0.55, edgecolor="none"
)

axes[1, 0].set_title("3. Cumulative Net Revenue by Sales Region", fontsize=12, fontweight="bold", loc="left")
axes[1, 0].set_ylabel("Total Net Revenue ($)", fontweight="medium")
axes[1, 0].yaxis.set_major_formatter(ticker.FuncFormatter(currency_k_formatter))
axes[1, 0].grid(True, axis='y', linestyle=":", alpha=0.6)

# Data labels on top of bars
for bar in bars:
    height = bar.get_height()
    axes[1, 0].annotate(
        f"${height * 1e-3:.1f}K",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4), textcoords="offset points",
        ha='center', va='bottom', fontsize=9.5, fontweight="bold"
    )

# ------------------------------------------------------------------------------
# PANEL [1, 1]: BIVARIATE RELATIONSHIP (Discount Rate vs Profit Margin)
# ------------------------------------------------------------------------------
sns.regplot(
    data=df, x=df["Discount_Pct"] * 100, y="Profit_Margin_Pct",
    scatter_kws={"color": "#64748b", "alpha": 0.35, "s": 22},
    line_kws={"color": "#dc2626", "linewidth": 2.2},
    ax=axes[1, 1]
)

axes[1, 1].set_title("4. Impact of Discount Rate on Profit Margin", fontsize=12, fontweight="bold", loc="left")
axes[1, 1].set_xlabel("Discount Rate (%)", fontweight="medium")
axes[1, 1].set_ylabel("Profit Margin (%)", fontweight="medium")
axes[1, 1].xaxis.set_major_formatter(ticker.PercentFormatter())
axes[1, 1].yaxis.set_major_formatter(ticker.PercentFormatter())
axes[1, 1].grid(True, linestyle=":", alpha=0.6)

# Correlation Coefficient Annotation
r_val = np.corrcoef(df["Discount_Pct"], df["Profit_Margin_Pct"])[0, 1]
axes[1, 1].text(
    0.05, 0.15, f"Pearson r = {r_val:.2f} (Strong Inverse Relationship)",
    transform=axes[1, 1].transAxes,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
    fontweight="bold", color="#dc2626", fontsize=9.5
)

# ------------------------------------------------------------------------------
# STEP 2: POLISH & EXPORT
# ------------------------------------------------------------------------------
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Save high-resolution chart artifact
output_image = "commercial_operations_dashboard.png"
plt.savefig(output_image, dpi=300, bbox_inches="tight")
print(f" Dashboard exported successfully to '{output_image}' at 300 DPI.")

if __name__ == "__main__":
    plt.show()
