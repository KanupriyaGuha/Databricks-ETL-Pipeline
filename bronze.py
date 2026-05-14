"""
bronze.py
Bronze Layer — Raw ingestion into Delta tables.
Loads CSVs as-is with minimal schema enforcement.
No business logic, no cleaning — pure landing zone.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, BooleanType
)
from delta import configure_spark_with_delta_pip
import time
import os


def get_spark():
    builder = (
        SparkSession.builder
        .appName("EcommerceETL_Bronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


# ── Schemas ───────────────────────────────────────────────────────────────────
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",  StringType(),    False),
    StructField("signup_date",  TimestampType(), True),
    StructField("age",          IntegerType(),   True),
    StructField("region",       StringType(),    True),
    StructField("tier",         StringType(),    True),
    StructField("email",        StringType(),    True),
])

PRODUCTS_SCHEMA = StructType([
    StructField("product_id",  StringType(),  False),
    StructField("category",    StringType(),  True),
    StructField("unit_price",  DoubleType(),  True),
    StructField("is_active",   BooleanType(), True),
])

TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id",  StringType(),    False),
    StructField("customer_id",     StringType(),    True),
    StructField("product_id",      StringType(),    True),
    StructField("order_date",      TimestampType(), True),
    StructField("quantity",        IntegerType(),   True),
    StructField("unit_price",      DoubleType(),    True),
    StructField("order_amount",    DoubleType(),    True),
    StructField("payment_method",  StringType(),    True),
])

RETURNS_SCHEMA = StructType([
    StructField("return_id",      StringType(),    False),
    StructField("transaction_id", StringType(),    True),
    StructField("customer_id",    StringType(),    True),
    StructField("return_date",    TimestampType(), True),
    StructField("reason",         StringType(),    True),
    StructField("refund_amount",  DoubleType(),    True),
])


def ingest_table(spark, csv_path: str, schema, delta_path: str, table_name: str) -> dict:
    """Load one CSV → Delta table. Returns stats dict."""
    t0 = time.time()
    print(f"\n[Bronze] Ingesting {table_name}...")

    df = (
        spark.read
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(schema)
        .csv(csv_path)
    )

    # Add ingestion metadata
    df = df.withColumn("_ingested_at", F.current_timestamp()) \
           .withColumn("_source_file", F.lit(os.path.basename(csv_path)))

    row_count = df.count()

    # Write as Delta (overwrite for idempotency)
    df.write.format("delta").mode("overwrite").save(delta_path)

    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {row_count:,} rows → {delta_path} ({elapsed}s)")

    return {
        "table":     table_name,
        "layer":     "bronze",
        "rows":      row_count,
        "path":      delta_path,
        "elapsed_s": elapsed,
        "status":    "success",
    }


def run_bronze(data_dir: str = "data", delta_base: str = "delta/bronze") -> list:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    os.makedirs(delta_base, exist_ok=True)

    tables = [
        ("customers.csv",    CUSTOMERS_SCHEMA,    "customers"),
        ("products.csv",     PRODUCTS_SCHEMA,     "products"),
        ("transactions.csv", TRANSACTIONS_SCHEMA, "transactions"),
        ("returns.csv",      RETURNS_SCHEMA,      "returns"),
    ]

    stats = []
    total_start = time.time()

    print("=" * 55)
    print("  BRONZE LAYER — Raw Ingestion")
    print("=" * 55)

    for fname, schema, tname in tables:
        csv_path   = os.path.join(data_dir, fname)
        delta_path = os.path.join(delta_base, tname)
        result     = ingest_table(spark, csv_path, schema, delta_path, tname)
        stats.append(result)

    total_rows = sum(s["rows"] for s in stats)
    total_time = round(time.time() - total_start, 2)

    print(f"\n{'=' * 55}")
    print(f"  Bronze complete: {total_rows:,} rows in {total_time}s")
    print(f"{'=' * 55}")

    spark.stop()
    return stats


if __name__ == "__main__":
    stats = run_bronze()
    for s in stats:
        print(f"  {s['table']:<15} {s['rows']:>10,} rows  {s['elapsed_s']}s")
