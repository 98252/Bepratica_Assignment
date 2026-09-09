# ==============================================================================
# STUDENT LAB ASSIGNMENT: DATA CLEANING, ENRICHMENT & AGGREGATION PIPELINE
# ==============================================================================
# This script executes the entire data engineering and analysis pipeline:
# - Part 1: Dataset Generation & Setup
# - Part 2: Trainer Baseline Fixes (Issues 1 - 3)
# - Part 3: Student Lab Tasks (Issues 4 - 9)
# - Part 4: Business Insights & Aggregations
# ==============================================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# STEP 0: ENVIRONMENT SETUP & DATA GENERATION
# ------------------------------------------------------------------------------
print("=" * 80)
print("STEP 0: GENERATING RAW DATASETS")
print("=" * 80)

messy_data = """Order_ID,Order_Date,Cust_ID,Customer_Name,Product_SKU,Units,Unit_Price,Shipping_City,Payment_Status
ORD-101,2024/01/15,C-1001,  Alice Smith ,SKU-E100,2,$250.00,New York,COMPLETED
ORD-102,16-01-2024,C-1002,Bob Jones,SKU-C200,1,$45.50,los angeles,Completed
ORD-103,2024-01-18,C-1003,charlie brown,SKU-B300,-3,$15.00,Chicago,REFUNDED
ORD-104,2024-01-20,C-1004,David Wilson,SKU-E100,5,$250.00,HOUSTON,Completed
ORD-105,2024.01.22,C-1005,Emma Watson,SKU-H400,1,unknown,Miami,PENDING
ORD-106,2024-01-25,C-1006,Frank Castle,SKU-C200,2,$45.50,New York,COMPLETED
ORD-107,2024-01-28,C-1007,Grace Hopper,SKU-E100,1,$250.00,CHICAGO,Completed
ORD-108,2024-02-01,C-1008,Henry Ford,SKU-B300,10,$15.00,Los Angeles,COMPLETED
ORD-109,2024-02-03,C-1009,Ivy League,SKU-H400,2,$120.00,Miami,PENDING
ORD-110,2024-02-05,C-1010,Jack Ryan,SKU-E100,1,$250.00,New York,COMPLETED
ORD-104,2024-01-20,C-1004,David Wilson,SKU-E100,5,$250.00,HOUSTON,Completed
ORD-111,2024-02-10,C-1011,Karen Page,SKU-C200,1,$45.50,Houston,COMPLETED
ORD-112,2024-02-12,C-1012,Leo Messi,SKU-B300,1,$15.00,New York,Completed
ORD-113,2024-02-15,C-1013,Mona Lisa,SKU-E100,500,$250.00,Miami,COMPLETED
ORD-114,2024-02-18,C-1014,Nick Fury,SKU-H400,3,$120.00,Los Angeles,FAILED
ORD-115,2024-02-20,C-1015,Olivia Pope,SKU-C200,2,$45.50,Chicago,COMPLETED
ORD-116,2024-02-22,C-1016,Peter Parker,SKU-B300,4,$15.00,New York,COMPLETED
ORD-117,2024-02-25,C-1017,Quinn Fabray,SKU-E100,1,$250.00,Houston,Completed
ORD-118,2024-02-28,C-1018,Rachel Green,SKU-H400,1,$120.00,Miami,COMPLETED
ORD-119,2024-03-02,C-1019,Steve Rogers,SKU-C200,2,$45.50,Los Angeles,COMPLETED
ORD-120,,C-1020,Tony Stark,SKU-E100,4,$250.00,New York,COMPLETED
"""

dim_products_data = """Product_SKU,Category,Cost_Price,Supplier
SKU-E100,Electronics,180.00,Apex Tech
SKU-C200,Clothing,22.00,Global Fabrics
SKU-B300,Books,8.50,ReadWell Press
SKU-H400,Home & Kitchen,75.00,HomeCrafters
SKU-X900,Toys,12.00,ToyVerse
"""

with open("raw_customer_orders.csv", "w") as f:
    f.write(messy_data.strip())

with open("dim_products.csv", "w") as f:
    f.write(dim_products_data.strip())

print("Files 'raw_customer_orders.csv' and 'dim_products.csv' generated.")

# ------------------------------------------------------------------------------
# STEP 1: TRAINER FIXES (ISSUES 1 - 3)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 1: RESOLVING ISSUES 1 - 3 (TRAINER BASELINE)")
print("=" * 80)

raw_orders = pd.read_csv("raw_customer_orders.csv")

# Issue 1: Deduplication (Keep first occurrence)
clean_orders = raw_orders.drop_duplicates(subset=['Order_ID'], keep='first').copy()

# Issue 2: Dirty Currency String Data Types (Unit_Price)
clean_orders['Unit_Price'] = (
    clean_orders['Unit_Price']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.strip()
)
clean_orders['Unit_Price'] = pd.to_numeric(clean_orders['Unit_Price'], errors='coerce')

sku_median_price = clean_orders.groupby('Product_SKU')['Unit_Price'].transform('median')
clean_orders['Unit_Price'] = clean_orders['Unit_Price'].fillna(sku_median_price)

# Issue 3: Inconsistent Date Formats & Missing Dates
clean_orders['Order_Date'] = pd.to_datetime(clean_orders['Order_Date'], errors='coerce', format='mixed')
clean_orders['Order_Date'] = clean_orders['Order_Date'].ffill()

print(f"Clean orders base records: {len(clean_orders)}")
print(clean_orders[['Order_ID', 'Order_Date', 'Cust_ID', 'Units', 'Unit_Price']].head(4))

# ------------------------------------------------------------------------------
# STEP 2: STUDENT LAB ASSIGNMENT (ISSUES 4 - 9)
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: RESOLVING ISSUES 4 - 9 (STUDENT LAB IMPLEMENTATION)")
print("=" * 80)

# Work on a dedicated copy to prevent unintended side effects
final_analytical_df = clean_orders.copy()

# --- ISSUE 4: NEGATIVE QUANTITIES ---
# Context: ORD-103 had Units = -3 due to a returns/logistics entry error.
# Action: Convert all Units values to absolute integers using .abs().
final_analytical_df['Units'] = final_analytical_df['Units'].abs()
print(" Issue 4 resolved: Negative quantities converted to absolute values.")

# --- ISSUE 5: GROSS OUTLIERS ---
# Context: ORD-113 had Units = 500 (human typo for 5).
# Action: Replace gross typo values (> 20) with the intended 5, or cap by threshold.
final_analytical_df['Units'] = np.where(final_analytical_df['Units'] > 20, 5, final_analytical_df['Units'])
print(" Issue 5 resolved: Extreme outlier in Units treated (500 replaced with 5).")

# --- ISSUE 6: STRING STANDARDISATION ---
# Context: Inconsistent capitalization ('los angeles', 'HOUSTON') and padding spaces.
# Action:
#   - 'Customer_Name': Trim leading/trailing whitespace and convert to Title Case.
#   - 'Shipping_City': Trim leading/trailing whitespace and convert to Title Case.
#   - 'Payment_Status': Strip whitespace and convert to UPPERCASE.
final_analytical_df['Customer_Name'] = final_analytical_df['Customer_Name'].astype(str).str.strip().str.title()
final_analytical_df['Shipping_City'] = final_analytical_df['Shipping_City'].astype(str).str.strip().str.title()
final_analytical_df['Payment_Status'] = final_analytical_df['Payment_Status'].astype(str).str.strip().str.upper()
print(" Issue 6 resolved: String columns standardized (Title Case & Uppercase).")

# --- ISSUE 7: DATE FEATURE ENGINEERING ---
# Context: Derive temporal dimensions for business cadence and weekend behavioral analysis.
# Action:
#   - 'Order_Month': Integer month (1 to 12) via .dt.month.
#   - 'Order_DayName': Day of the week name ('Monday', 'Tuesday', etc.) via .dt.day_name().
#   - 'Is_Weekend': Boolean flag indicating Saturday (5) or Sunday (6) via .dt.dayofweek.isin([5, 6]).
final_analytical_df['Order_Month'] = final_analytical_df['Order_Date'].dt.month
final_analytical_df['Order_DayName'] = final_analytical_df['Order_Date'].dt.day_name()
final_analytical_df['Is_Weekend'] = final_analytical_df['Order_Date'].dt.dayofweek.isin([5, 6])
print(" Issue 7 resolved: Engineered date features (Order_Month, Order_DayName, Is_Weekend).")

# --- ISSUE 8: RELATIONAL ENRICHMENT ---
# Context: Join transaction orders with product catalog to compute financial margins.
# Action:
#   - Left merge with dim_products.csv on 'Product_SKU'.
#   - Calculate 'Gross_Revenue' = Units * Unit_Price.
#   - Calculate 'Net_Profit' = (Unit_Price - Cost_Price) * Units.
dim_products_df = pd.read_csv("dim_products.csv")

final_analytical_df = pd.merge(
    final_analytical_df,
    dim_products_df[['Product_SKU', 'Category', 'Cost_Price', 'Supplier']],
    on='Product_SKU',
    how='left'
)

final_analytical_df['Gross_Revenue'] = (final_analytical_df['Units'] * final_analytical_df['Unit_Price']).round(2)
final_analytical_df['Net_Profit'] = ((final_analytical_df['Unit_Price'] - final_analytical_df['Cost_Price']) * final_analytical_df['Units']).round(2)
print(" Issue 8 resolved: Merged with dim_products and calculated Gross_Revenue & Net_Profit.")

# --- ISSUE 9: CROSS-TABULATION ANALYSIS ---
# Context: Executive breakdown of Net Profit across product categories and shipping destinations.
# Action:
#   - Pivot table with Category (rows), Shipping_City (columns), and sum of Net_Profit.
#   - margins=True provides Total/Grand Total columns and rows.
profit_pivot_table = final_analytical_df.pivot_table(
    index='Category',
    columns='Shipping_City',
    values='Net_Profit',
    aggfunc='sum',
    fill_value=0.0,
    margins=True,
    margins_name='Total'
).round(2)

print(" Issue 9 resolved: Cross-tabulation pivot table generated.")

# ------------------------------------------------------------------------------
# STEP 3: DISPLAY RESULTS & VERIFICATION
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL ANALYTICAL DATAFRAME SUMMARY (Shape: {})".format(final_analytical_df.shape))
print("=" * 80)
print(final_analytical_df[[
    'Order_ID', 'Customer_Name', 'Product_SKU', 'Category', 
    'Units', 'Unit_Price', 'Cost_Price', 'Gross_Revenue', 'Net_Profit', 
    'Shipping_City', 'Payment_Status', 'Order_DayName', 'Is_Weekend'
]].to_string(index=False))

print("\n" + "=" * 80)
print("EXECUTIVE PIVOT TABLE: NET PROFIT BY CATEGORY & SHIPPING CITY")
print("=" * 80)
print(profit_pivot_table)

# Export analytical dataframe for downstream BI / warehouse ingest
final_analytical_df.to_csv("final_analytical_orders.csv", index=False)
print("\n Exported analysis-ready dataset to 'final_analytical_orders.csv'.")
