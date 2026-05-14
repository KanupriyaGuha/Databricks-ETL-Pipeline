"""
generate_data.py
Synthetic e-commerce data generator for Databricks ETL Pipeline.
Generates 4 tables: customers, products, transactions, returns
Total volume: ~1M+ rows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
N_CUSTOMERS    = 100_000
N_PRODUCTS     = 500
N_TRANSACTIONS = 1_000_000
RETURN_RATE    = 0.08       # 8% of transactions become returns
START_DATE     = datetime(2022, 1, 1)
END_DATE       = datetime(2024, 12, 31)
TOTAL_DAYS     = (END_DATE - START_DATE).days

# ── Reference Data ────────────────────────────────────────────────────────────
REGIONS     = ["North", "South", "East", "West", "Central"]
TIERS       = ["Bronze", "Silver", "Gold", "Platinum"]
TIER_WEIGHTS= [0.50,     0.30,    0.15,   0.05]

CATEGORIES  = [
    "Electronics", "Clothing", "Home & Garden", "Sports", "Beauty",
    "Books", "Toys", "Automotive", "Food & Grocery", "Office Supplies"
]

RETURN_REASONS = [
    "Defective product", "Wrong item received", "Changed mind",
    "Does not fit", "Better price found", "Item not as described",
    "Arrived too late", "Duplicate order"
]


def generate_customers() -> pd.DataFrame:
    print("Generating customers...")
    customer_ids = [f"C{i:07d}" for i in range(1, N_CUSTOMERS + 1)]
    signup_days  = np.random.randint(0, TOTAL_DAYS - 30, N_CUSTOMERS)
    signup_dates = [START_DATE + timedelta(days=int(d)) for d in signup_days]

    tiers   = np.random.choice(TIERS, N_CUSTOMERS, p=TIER_WEIGHTS)
    regions = np.random.choice(REGIONS, N_CUSTOMERS)
    ages    = np.random.randint(18, 75, N_CUSTOMERS)

    # Email domain mix
    domains = np.random.choice(
        ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"],
        N_CUSTOMERS, p=[0.45, 0.20, 0.18, 0.10, 0.07]
    )

    # ~2% null emails, ~1% null regions (dirty data for Silver to fix)
    emails = [
        f"user{i}@{domains[i]}" if random.random() > 0.02 else None
        for i in range(N_CUSTOMERS)
    ]
    region_col = [
        r if random.random() > 0.01 else None
        for r in regions
    ]

    df = pd.DataFrame({
        "customer_id":  customer_ids,
        "signup_date":  signup_dates,
        "age":          ages,
        "region":       region_col,
        "tier":         tiers,
        "email":        emails,
    })
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    print(f"  → {len(df):,} customers")
    return df


def generate_products() -> pd.DataFrame:
    print("Generating products...")
    product_ids = [f"P{i:05d}" for i in range(1, N_PRODUCTS + 1)]
    categories  = np.random.choice(CATEGORIES, N_PRODUCTS)

    # Price ranges differ by category
    price_map = {
        "Electronics":      (49,  1200),
        "Clothing":         (12,   180),
        "Home & Garden":    (15,   350),
        "Sports":           (20,   400),
        "Beauty":           (8,    120),
        "Books":            (5,     60),
        "Toys":             (10,   150),
        "Automotive":       (25,   600),
        "Food & Grocery":   (3,     80),
        "Office Supplies":  (5,    200),
    }

    prices = [
        round(random.uniform(*price_map[c]), 2)
        for c in categories
    ]

    # ~1% discontinued products
    is_active = [random.random() > 0.01 for _ in range(N_PRODUCTS)]

    df = pd.DataFrame({
        "product_id":   product_ids,
        "category":     categories,
        "unit_price":   prices,
        "is_active":    is_active,
    })
    print(f"  → {len(df):,} products")
    return df


def generate_transactions(customers: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    print("Generating transactions (this takes ~30 seconds)...")

    cust_ids    = customers["customer_id"].values
    cust_dates  = pd.to_datetime(customers["signup_date"]).values
    prod_ids    = products["product_id"].values
    prod_prices = products["unit_price"].values

    transaction_ids = [f"T{i:09d}" for i in range(1, N_TRANSACTIONS + 1)]

    # Random customer for each transaction
    cust_idx = np.random.randint(0, N_CUSTOMERS, N_TRANSACTIONS)
    selected_custs  = cust_ids[cust_idx]
    selected_signup = cust_dates[cust_idx]

    # Order date = after signup date, within dataset range
    end_ts    = np.datetime64(END_DATE)
    rand_days = np.random.randint(0, TOTAL_DAYS, N_TRANSACTIONS)
    order_ts  = np.array([np.datetime64(START_DATE)] * N_TRANSACTIONS) + rand_days.astype("timedelta64[D]")

    # Ensure order date >= signup date
    order_ts  = np.where(order_ts < selected_signup, selected_signup, order_ts)
    order_ts  = np.where(order_ts > end_ts, end_ts, order_ts)

    # Random product + quantity
    prod_idx   = np.random.randint(0, N_PRODUCTS, N_TRANSACTIONS)
    quantities = np.random.randint(1, 6, N_TRANSACTIONS)
    unit_prices = prod_prices[prod_idx]

    # Price with ±5% variance
    variance    = np.random.uniform(0.95, 1.05, N_TRANSACTIONS)
    order_amts  = (unit_prices * quantities * variance).round(2)

    # Payment methods
    pay_methods = np.random.choice(
        ["credit_card", "debit_card", "paypal", "apple_pay", "bank_transfer"],
        N_TRANSACTIONS, p=[0.40, 0.25, 0.20, 0.10, 0.05]
    )

    # ~0.5% null payment method (dirty data)
    null_mask = np.random.random(N_TRANSACTIONS) < 0.005
    pay_methods = pay_methods.astype(object)
    pay_methods[null_mask] = None

    # ~0.3% negative amounts (bad data for Silver to flag)
    bad_mask = np.random.random(N_TRANSACTIONS) < 0.003
    order_amts[bad_mask] = -order_amts[bad_mask]

    df = pd.DataFrame({
        "transaction_id":  transaction_ids,
        "customer_id":     selected_custs,
        "product_id":      prod_ids[prod_idx],
        "order_date":      order_ts,
        "quantity":        quantities,
        "unit_price":      unit_prices.round(2),
        "order_amount":    order_amts,
        "payment_method":  pay_methods,
    })
    df["order_date"] = pd.to_datetime(df["order_date"])
    print(f"  → {len(df):,} transactions | ${df[df['order_amount']>0]['order_amount'].sum():,.0f} gross revenue")
    return df


def generate_returns(transactions: pd.DataFrame) -> pd.DataFrame:
    print("Generating returns...")

    # Only positive-amount transactions can be returned
    valid_tx = transactions[transactions["order_amount"] > 0].copy()
    n_returns = int(len(valid_tx) * RETURN_RATE)

    sampled = valid_tx.sample(n=n_returns, random_state=SEED)

    return_ids = [f"R{i:08d}" for i in range(1, n_returns + 1)]

    # Return date = 1–30 days after order date
    return_delays = np.random.randint(1, 31, n_returns)
    return_dates  = pd.to_datetime(sampled["order_date"].values) + pd.to_timedelta(return_delays, unit="D")
    return_dates  = return_dates.where(return_dates <= pd.Timestamp(END_DATE), pd.Timestamp(END_DATE))

    reasons       = np.random.choice(RETURN_REASONS, n_returns)
    refund_amts   = (sampled["order_amount"].values * np.random.uniform(0.80, 1.0, n_returns)).round(2)

    df = pd.DataFrame({
        "return_id":      return_ids,
        "transaction_id": sampled["transaction_id"].values,
        "customer_id":    sampled["customer_id"].values,
        "return_date":    return_dates,
        "reason":         reasons,
        "refund_amount":  refund_amts,
    })
    df["return_date"] = pd.to_datetime(df["return_date"])
    print(f"  → {len(df):,} returns | ${df['refund_amount'].sum():,.0f} refunded")
    return df


def save_all(output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    customers    = generate_customers()
    products     = generate_products()
    transactions = generate_transactions(customers, products)
    returns      = generate_returns(transactions)

    # Format all datetime columns explicitly so Spark reads them correctly
    customers["signup_date"]    = customers["signup_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    transactions["order_date"]  = pd.to_datetime(transactions["order_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    returns["return_date"]      = pd.to_datetime(returns["return_date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    customers.to_csv(f"{output_dir}/customers.csv",       index=False)
    products.to_csv(f"{output_dir}/products.csv",         index=False)
    transactions.to_csv(f"{output_dir}/transactions.csv", index=False)
    returns.to_csv(f"{output_dir}/returns.csv",           index=False)

    total_rows = len(customers) + len(products) + len(transactions) + len(returns)
    print(f"\n✅ All data saved to {output_dir}/")
    print(f"   customers.csv    → {len(customers):,} rows")
    print(f"   products.csv     → {len(products):,} rows")
    print(f"   transactions.csv → {len(transactions):,} rows")
    print(f"   returns.csv      → {len(returns):,} rows")
    print(f"   Total            → {total_rows:,} rows")

    return customers, products, transactions, returns


if __name__ == "__main__":
    save_all("data")
