"""Generates a deliberately messy orders dataset to test the auditor."""
import numpy as np, pandas as pd

rng = np.random.default_rng(7)
n = 1000
cities = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Hyderabad"]
df = pd.DataFrame({
    "order_id": range(1, n + 1),
    "customer_name": [f"Customer {i % 300}" for i in range(n)],
    "city": rng.choice(cities, n),
    "order_date": pd.date_range("2024-01-01", periods=n, freq="8h").strftime("%Y-%m-%d"),
    "quantity": rng.integers(1, 10, n),
    "amount": rng.normal(2500, 600, n).round(2).astype(str),
    "email": [f"user{i}@example.com" for i in range(n)],
    "currency": "INR",
})

# --- inject problems ---
df.loc[rng.choice(n, 40, replace=False), "city"] = "bangalore"
df.loc[rng.choice(n, 25, replace=False), "city"] = "Bengaluru "
df.loc[rng.choice(n, 20, replace=False), "city"] = "MUMBAI"
df.loc[rng.choice(n, 60, replace=False), "email"] = None
df.loc[rng.choice(n, 30, replace=False), "email"] = "N/A"
df.loc[rng.choice(n, 15, replace=False), "order_date"] = "not recorded"
df.loc[rng.choice(n, 12, replace=False), "order_date"] = "31/02/2024"
df.loc[rng.choice(n, 35, replace=False), "amount"] = "N/A"
df.loc[rng.choice(n, 20, replace=False), "amount"] = "₹1,999"
df.loc[rng.choice(n, 6, replace=False), "quantity"] = 900
df.loc[rng.choice(n, 4, replace=False), "quantity"] = -3
df.loc[rng.choice(n, 25, replace=False), "customer_name"] += "  "
df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)  # duplicate rows
df.to_csv("messy_orders.csv", index=False)
print(f"wrote messy_orders.csv: {len(df)} rows")
