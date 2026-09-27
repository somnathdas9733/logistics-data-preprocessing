"""Logistics Data Collection, Cleaning, and Preprocessing Pipeline Week 2

Task - Logistics Data Analyst Intern.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==============================================================================
# STEP 1: SIMULATE A DIRTY LOGISTICS DATASET (Data Collection Simulation)
# ==============================================================================
np.random.seed(42)
n_records = 1000

# Base realistic data
order_ids = [f"ORD-{10000 + i}" for i in range(n_records)]
order_dates = pd.date_range(
    start="2026-01-01", periods=n_records, freq="h"
).astype(str)
scheduled_days = np.random.choice([2, 3, 5, 7], size=n_records)
actual_days = scheduled_days + np.random.normal(0.5, 1.2, size=n_records)
shipping_costs = np.random.uniform(15.0, 180.0, size=n_records)
package_weights = np.random.exponential(scale=12.0, size=n_records) + 0.5
distance_km = np.random.uniform(50.0, 2400.0, size=n_records)
shipping_modes = np.random.choice(
    ["Standard Class", "Second Class", "First Class", "Same Day"],
    size=n_records,
)
delivery_statuses = np.random.choice(
    ["On-Time", "Late", "Canceled"], size=n_records, p=[0.70, 0.25, 0.05]
)

df = pd.DataFrame(
    {
        "Order_Id": order_ids,
        "Order_Date": order_dates,
        "Scheduled_Days": scheduled_days,
        "Actual_Days": actual_days,
        "Shipping_Cost": shipping_costs,
        "Package_Weight_kg": package_weights,
        "Distance_km": distance_km,
        "Shipping_Mode": shipping_modes,
        "Delivery_Status": delivery_statuses,
    }
)

# Inject common logistics real-world dirty data patterns:
# 1. Missing values
df.loc[np.random.choice(n_records, 40, replace=False), "Actual_Days"] = np.nan
df.loc[np.random.choice(n_records, 25, replace=False), "Shipping_Cost"] = np.nan
df.loc[np.random.choice(n_records, 15, replace=False), "Shipping_Mode"] = np.nan
df.loc[np.random.choice(n_records, 5, replace=False), "Delivery_Status"] = np.nan

# 2. Impossible outliers & clock desync
df.loc[5, "Actual_Days"] = -4.0  # Impossible negative transit days
df.loc[12, "Actual_Days"] = 380.0  # Lost/abandoned system artifact
df.loc[22, "Shipping_Cost"] = -25.0  # Invoicing glitch
df.loc[35, "Shipping_Cost"] = 2400.0  # Data entry extreme spike

# 3. Categorical white-spaces and inconsistent casing
df.loc[10:15, "Shipping_Mode"] = "  standard class "

print("--- RAW DATA AUDIT ---")
print("Null counts:\n", df.isnull().sum())
print("\nInitial Descriptive Stats:\n", df.describe())

# ==============================================================================
# STEP 2: DATA CLEANING & IMPUTATION
# ==============================================================================

# A. Standardize column names
df.columns = df.columns.str.strip().str.lower()

# B. Drop rows missing critical IDs or target variable
df.dropna(subset=["order_id", "delivery_status"], inplace=True)

# C. Clean categorical strings
df["shipping_mode"] = df["shipping_mode"].astype(str).str.strip().str.title()
df.loc[df["shipping_mode"] == "Nan", "shipping_mode"] = "Standard Class"

# D. Median imputation for numerical fields (resilient to skewness)
median_actual_days = df["actual_days"].median()
df["actual_days"] = df["actual_days"].fillna(median_actual_days)

median_cost = df["shipping_cost"].median()
df["shipping_cost"] = df["shipping_cost"].fillna(median_cost)

# ==============================================================================
# STEP 3: OUTLIER DETECTION & IQR CAPPING (WINSORIZATION)
# ==============================================================================

# Outlier remediation for transit days
q1_days = df["actual_days"].quantile(0.25)
q3_days = df["actual_days"].quantile(0.75)
iqr_days = q3_days - q1_days

lower_bound_days = max(
    0.0, q1_days - 1.5 * iqr_days
)  # Transit days cannot be negative
upper_bound_days = q3_days + 1.5 * iqr_days

# Cap extreme values to IQR limits rather than dropping records
df["actual_days_cleaned"] = np.clip(
    df["actual_days"], lower_bound_days, upper_bound_days
)

# Fix negative shipping costs to median
df.loc[df["shipping_cost"] < 0, "shipping_cost"] = median_cost

# Capping extreme shipping cost spikes
q1_cost = df["shipping_cost"].quantile(0.25)
q3_cost = df["shipping_cost"].quantile(0.75)
iqr_cost = q3_cost - q1_cost
upper_bound_cost = q3_cost + 1.5 * iqr_cost
df["shipping_cost_cleaned"] = np.clip(df["shipping_cost"], 0, upper_bound_cost)

# ==============================================================================
# STEP 4: FEATURE ENGINEERING
# ==============================================================================
df["order_date"] = pd.to_datetime(df["order_date"])
df["sla_delay_days"] = df["actual_days_cleaned"] - df["scheduled_days"]
df["is_sla_breached"] = np.where(df["sla_delay_days"] > 0, 1, 0)

# ==============================================================================
# STEP 5: NORMALIZATION & SCALING
# ==============================================================================

# Standard Scaler (Z-Score) for normally distributed metrics
scaler = StandardScaler()
df[["scaled_distance", "scaled_cost"]] = scaler.fit_transform(
    df[["distance_km", "shipping_cost_cleaned"]]
)

# Min-Max Scaler (0 to 1) for bounded operational metrics
minmax = MinMaxScaler()
df["normalized_weight"] = minmax.fit_transform(df[["package_weight_kg"]])

print("\n--- CLEANED DATA AUDIT ---")
print(f"Cleaned Records Count: {len(df)}")
print(
    "Actual Days Range (Post-Clean):",
    df["actual_days_cleaned"].min(),
    "to",
    df["actual_days_cleaned"].max(),
)
print("Remaining Nulls:", df.isnull().sum().sum())

# ==============================================================================
# STEP 6: EXPORT CLEANED ARTIFACT
# ==============================================================================
df.to_csv("logistics_data_cleaned.csv", index=False)
print(
    "\nSuccessfully exported cleaned logistics data to 'logistics_data_cleaned.csv'."
)