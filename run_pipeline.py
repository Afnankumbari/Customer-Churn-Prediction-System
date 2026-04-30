"""
run_pipeline.py
---------------
One-shot script to run the full pipeline:
  1. Generate dataset
  2. Run EDA
  3. Train models & save best
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)


def main():
    print("=" * 60)
    print("  ChurnShield — Full Pipeline Runner")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/3] Generating dataset...")
    from generate_data import generate_dataset, write_sql
    data_dir = os.path.join(ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_dataset()
    csv_path = os.path.join(data_dir, "customers.csv")
    df.to_csv(csv_path, index=False)
    write_sql(df, data_dir)
    print(f"  ✅ {len(df):,} customers | Churn rate: {df['Churn'].mean():.1%}")

    # Step 2: EDA
    print("\n[2/3] Running EDA...")
    from eda import run_eda
    import pandas as pd
    df_loaded = pd.read_csv(csv_path)
    run_eda(df_loaded, os.path.join(ROOT, "assets"))

    # Step 3: Train
    print("\n[3/3] Training models...")
    from train import train_and_evaluate
    train_and_evaluate(
        csv_path=csv_path,
        model_out=os.path.join(ROOT, "models", "best_model.pkl"),
        assets_dir=os.path.join(ROOT, "assets"),
    )

    print("\n" + "=" * 60)
    print("  ✅ Pipeline complete!")
    print(f"  Data   → data/customers.csv")
    print(f"  SQL    → data/schema.sql, data/inserts.sql")
    print(f"  Plots  → assets/")
    print(f"  Model  → models/best_model.pkl")
    print("\n  Launch dashboard:")
    print("    python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
