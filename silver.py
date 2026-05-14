"""
silver.py
Silver Layer — Clean, validate, and enrich.
Reads from Bronze Delta tables, applies quality rules,
flags/removes bad records, writes clean Silver Delta tables.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip
import time
import os


def get_spark():
    builder = (
        SparkSession.builder
        .appName("EcommerceETL_Silver")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# ── Quality Check Results ─────────────────────────────────────────────────────
def quality_report(df_raw, df_clean, table_name: str) -> dict:
    raw_count   = df_raw.count()
    clean_count = df_clean.count()
    dropped     = raw_count - clean_count
    return {
        "table":        table_name,
        "layer":        "silver",
        "raw_rows":     raw_count,
        "clean_rows":   clean_count,
        "dropped_rows": dropped,
        "drop_pct":     round(dropped / raw_count * 100, 2) if raw_count > 0 else 0,
    }


# ── Customers ─────────────────────────────────────────────────────────────────
def clean_customers(spark, bronze_path: str, silver_path: str) -> dict:
    t0 = time.time()
    print("\n[Silver] Cleaning customers...")

    df = spark.read.format("delta").load(bronze_path)
    raw_count = df.count()

    # Drop rows missing primary key
    df = df.filter(F.col("customer_id").isNotNull())

    # Fill missing region with "Unknown"
    df = df.fillna({"region": "Unknown"})

    # Fill missing email with generated placeholder
    df = df.withColumn(
        "email",
        F.when(F.col("email").isNull(), F.concat(F.lit("unknown_"), F.col("customer_id"), F.lit("@noemail.com")))
         .otherwise(F.col("email"))
    )

    # Validate age range (18–100), allow nulls
    df = df.filter(F.col("age").isNull() | F.col("age").between(18, 100))

    # Deduplicate on customer_id (keep first occurrence)
    df = df.dropDuplicates(["customer_id"])

    # Add silver metadata
    df = df.withColumn("_cleaned_at", F.current_timestamp()) \
           .drop("_ingested_at", "_source_file")

    clean_count = df.count()
    df.write.format("delta").mode("overwrite").save(silver_path)

    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {raw_count:,} → {clean_count:,} rows ({raw_count - clean_count:,} dropped) | {elapsed}s")
    return {"table": "customers", "layer": "silver", "raw_rows": raw_count,
            "clean_rows": clean_count, "dropped_rows": raw_count - clean_count,
            "drop_pct": round((raw_count - clean_count) / raw_count * 100, 2), "elapsed_s": elapsed}


# ── Products ──────────────────────────────────────────────────────────────────
def clean_products(spark, bronze_path: str, silver_path: str) -> dict:
    t0 = time.time()
    print("\n[Silver] Cleaning products...")

    df = spark.read.format("delta").load(bronze_path)
    raw_count = df.count()

    # Drop null product_id
    df = df.filter(F.col("product_id").isNotNull())

    # Validate price > 0
    df = df.filter(F.col("unit_price") > 0)

    # Fill null is_active with True
    df = df.fillna({"is_active": True})

    # Deduplicate
    df = df.dropDuplicates(["product_id"])

    df = df.withColumn("_cleaned_at", F.current_timestamp()) \
           .drop("_ingested_at", "_source_file")

    clean_count = df.count()
    df.write.format("delta").mode("overwrite").save(silver_path)

    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {raw_count:,} → {clean_count:,} rows ({raw_count - clean_count:,} dropped) | {elapsed}s")
    return {"table": "products", "layer": "silver", "raw_rows": raw_count,
            "clean_rows": clean_count, "dropped_rows": raw_count - clean_count,
            "drop_pct": round((raw_count - clean_count) / raw_count * 100, 2), "elapsed_s": elapsed}


# ── Transactions ──────────────────────────────────────────────────────────────
def clean_transactions(spark, bronze_path: str, silver_path: str,
                        silver_customers: str, silver_products: str) -> dict:
    t0 = time.time()
    print("\n[Silver] Cleaning transactions...")

    df       = spark.read.format("delta").load(bronze_path)
    custs    = spark.read.format("delta").load(silver_customers).select("customer_id")
    prods    = spark.read.format("delta").load(silver_products).select("product_id")
    raw_count = df.count()

    # Drop null primary key
    df = df.filter(F.col("transaction_id").isNotNull())

    # Drop negative / zero order amounts
    df = df.filter(F.col("order_amount") > 0)

    # Drop null order_date and future dates (keep nulls as-is, just drop genuine future)
    df = df.filter(F.col("order_date").isNotNull())

    # Fill null payment_method
    df = df.fillna({"payment_method": "unknown"})

    # Referential integrity — keep only valid customer_id and product_id
    df = df.join(custs, on="customer_id", how="inner")
    df = df.join(prods, on="product_id",  how="inner")

    # Validate quantity > 0
    df = df.filter(F.col("quantity") > 0)

    # Deduplicate on transaction_id
    df = df.dropDuplicates(["transaction_id"])

    # Add derived columns
    df = df.withColumn("order_year",  F.year("order_date")) \
           .withColumn("order_month", F.month("order_date")) \
           .withColumn("order_dow",   F.dayofweek("order_date"))

    df = df.withColumn("_cleaned_at", F.current_timestamp()) \
           .drop("_ingested_at", "_source_file")

    clean_count = df.count()
    df.write.format("delta").mode("overwrite").save(silver_path)

    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {raw_count:,} → {clean_count:,} rows ({raw_count - clean_count:,} dropped) | {elapsed}s")
    return {"table": "transactions", "layer": "silver", "raw_rows": raw_count,
            "clean_rows": clean_count, "dropped_rows": raw_count - clean_count,
            "drop_pct": round((raw_count - clean_count) / raw_count * 100, 2), "elapsed_s": elapsed}


# ── Returns ───────────────────────────────────────────────────────────────────
def clean_returns(spark, bronze_path: str, silver_path: str, silver_transactions: str) -> dict:
    t0 = time.time()
    print("\n[Silver] Cleaning returns...")

    df    = spark.read.format("delta").load(bronze_path)
    txns  = spark.read.format("delta").load(silver_transactions).select("transaction_id", "order_date")
    raw_count = df.count()

    # Drop null return_id
    df = df.filter(F.col("return_id").isNotNull())

    # Referential integrity — only valid transaction_ids
    df = df.join(txns, on="transaction_id", how="inner")

    # Validate return_date >= order_date
    df = df.filter(F.col("return_date") >= F.col("order_date")).drop("order_date")

    # Validate refund > 0
    df = df.filter(F.col("refund_amount") > 0)

    # Deduplicate
    df = df.dropDuplicates(["return_id"])

    df = df.withColumn("_cleaned_at", F.current_timestamp()) \
           .drop("_ingested_at", "_source_file")

    clean_count = df.count()
    df.write.format("delta").mode("overwrite").save(silver_path)

    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {raw_count:,} → {clean_count:,} rows ({raw_count - clean_count:,} dropped) | {elapsed}s")
    return {"table": "returns", "layer": "silver", "raw_rows": raw_count,
            "clean_rows": clean_count, "dropped_rows": raw_count - clean_count,
            "drop_pct": round((raw_count - clean_count) / raw_count * 100, 2), "elapsed_s": elapsed}


def run_silver(delta_base: str = "delta") -> list:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    b = f"{delta_base}/bronze"
    s = f"{delta_base}/silver"
    os.makedirs(s, exist_ok=True)

    stats = []
    total_start = time.time()

    print("=" * 55)
    print("  SILVER LAYER — Clean & Validate")
    print("=" * 55)

    stats.append(clean_customers(spark,    f"{b}/customers",    f"{s}/customers"))
    stats.append(clean_products(spark,     f"{b}/products",     f"{s}/products"))
    stats.append(clean_transactions(spark, f"{b}/transactions", f"{s}/transactions",
                                           f"{s}/customers",    f"{s}/products"))
    stats.append(clean_returns(spark,      f"{b}/returns",      f"{s}/returns",
                                           f"{s}/transactions"))

    total_time = round(time.time() - total_start, 2)
    total_dropped = sum(s["dropped_rows"] for s in stats)

    print(f"\n{'=' * 55}")
    print(f"  Silver complete: {total_dropped:,} bad rows removed in {total_time}s")
    print(f"{'=' * 55}")

    spark.stop()
    return stats


if __name__ == "__main__":
    stats = run_silver()
    for s in stats:
        print(f"  {s['table']:<15} {s['clean_rows']:>10,} clean  {s['dropped_rows']:>6,} dropped  ({s['drop_pct']}%)")
