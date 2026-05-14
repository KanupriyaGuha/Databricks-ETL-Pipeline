"""
gold.py
Gold Layer — Business-ready aggregations.
Reads from Silver Delta tables, produces 4 Gold tables:
  - customer_metrics   : one row per customer (LTV, churn risk, tier)
  - monthly_revenue    : monthly revenue/orders/AOV trends
  - category_performance : revenue & return rate by product category
  - cohort_retention   : monthly cohort retention rates
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
        .appName("EcommerceETL_Gold")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def build_customer_metrics(spark, silver_base: str, gold_path: str) -> dict:
    t0 = time.time()
    print("\n[Gold] Building customer_metrics...")

    txns  = spark.read.format("delta").load(f"{silver_base}/transactions")
    custs = spark.read.format("delta").load(f"{silver_base}/customers")
    rets  = spark.read.format("delta").load(f"{silver_base}/returns")

    # Snapshot date = max order date + 1 day
    snapshot = txns.agg(F.max("order_date")).collect()[0][0]

    # Customer-level transaction aggregates
    cust_tx = txns.groupBy("customer_id").agg(
        F.sum("order_amount").alias("total_revenue"),
        F.count("transaction_id").alias("total_orders"),
        F.avg("order_amount").alias("avg_order_value"),
        F.min("order_date").alias("first_order_date"),
        F.max("order_date").alias("last_order_date"),
    )

    # Recency in days
    cust_tx = cust_tx.withColumn(
        "recency_days",
        F.datediff(F.lit(snapshot), F.col("last_order_date"))
    )

    # Returns per customer
    cust_ret = rets.groupBy("customer_id").agg(
        F.count("return_id").alias("total_returns"),
        F.sum("refund_amount").alias("total_refunded"),
    )

    # Join with customer profile
    df = custs.join(cust_tx, on="customer_id", how="left") \
              .join(cust_ret, on="customer_id", how="left")

    df = df.fillna({
        "total_revenue": 0.0, "total_orders": 0,
        "avg_order_value": 0.0, "total_returns": 0, "total_refunded": 0.0,
        "recency_days": 9999,
    })

    # Net revenue after refunds
    df = df.withColumn("net_revenue", F.col("total_revenue") - F.col("total_refunded"))

    # Churn risk flag
    df = df.withColumn("churn_risk",
        F.when(F.col("recency_days") > 180, "High")
         .when(F.col("recency_days") > 90,  "Medium")
         .otherwise("Low")
    )

    # Customer lifetime (days from signup to last order)
    df = df.withColumn("lifetime_days",
        F.datediff(F.col("last_order_date"), F.col("signup_date"))
    )

    df = df.withColumn("_created_at", F.current_timestamp())

    count = df.count()
    df.write.format("delta").mode("overwrite").save(gold_path)
    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {count:,} customer records | {elapsed}s")
    return {"table": "customer_metrics", "layer": "gold", "rows": count, "elapsed_s": elapsed}


def build_monthly_revenue(spark, silver_base: str, gold_path: str) -> dict:
    t0 = time.time()
    print("\n[Gold] Building monthly_revenue...")

    txns = spark.read.format("delta").load(f"{silver_base}/transactions")
    rets = spark.read.format("delta").load(f"{silver_base}/returns")

    monthly_tx = txns.withColumn("month", F.date_format("order_date", "yyyy-MM")) \
                     .groupBy("month").agg(
                         F.sum("order_amount").alias("gross_revenue"),
                         F.count("transaction_id").alias("total_orders"),
                         F.countDistinct("customer_id").alias("unique_customers"),
                         F.avg("order_amount").alias("avg_order_value"),
                     )

    monthly_ret = rets.withColumn("month", F.date_format("return_date", "yyyy-MM")) \
                      .groupBy("month").agg(
                          F.sum("refund_amount").alias("total_refunds"),
                          F.count("return_id").alias("total_returns"),
                      )

    df = monthly_tx.join(monthly_ret, on="month", how="left") \
                   .fillna({"total_refunds": 0.0, "total_returns": 0})

    df = df.withColumn("net_revenue", F.col("gross_revenue") - F.col("total_refunds")) \
           .withColumn("return_rate", F.round(F.col("total_returns") / F.col("total_orders") * 100, 2)) \
           .orderBy("month")

    df = df.withColumn("_created_at", F.current_timestamp())

    count = df.count()
    df.write.format("delta").mode("overwrite").save(gold_path)
    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {count:,} monthly records | {elapsed}s")
    return {"table": "monthly_revenue", "layer": "gold", "rows": count, "elapsed_s": elapsed}


def build_category_performance(spark, silver_base: str, gold_path: str) -> dict:
    t0 = time.time()
    print("\n[Gold] Building category_performance...")

    txns  = spark.read.format("delta").load(f"{silver_base}/transactions")
    prods = spark.read.format("delta").load(f"{silver_base}/products").select("product_id", "category")
    rets  = spark.read.format("delta").load(f"{silver_base}/returns")

    tx_cat = txns.join(prods, on="product_id", how="left") \
                 .groupBy("category").agg(
                     F.sum("order_amount").alias("gross_revenue"),
                     F.count("transaction_id").alias("total_orders"),
                     F.countDistinct("customer_id").alias("unique_customers"),
                     F.avg("order_amount").alias("avg_order_value"),
                     F.sum("quantity").alias("total_units_sold"),
                 )

    ret_cat = rets.join(txns.select("transaction_id", "product_id"), on="transaction_id", how="left") \
                  .join(prods, on="product_id", how="left") \
                  .groupBy("category").agg(
                      F.count("return_id").alias("total_returns"),
                      F.sum("refund_amount").alias("total_refunded"),
                  )

    df = tx_cat.join(ret_cat, on="category", how="left") \
               .fillna({"total_returns": 0, "total_refunded": 0.0})

    df = df.withColumn("net_revenue", F.col("gross_revenue") - F.col("total_refunded")) \
           .withColumn("return_rate", F.round(F.col("total_returns") / F.col("total_orders") * 100, 2)) \
           .orderBy(F.col("gross_revenue").desc())

    df = df.withColumn("_created_at", F.current_timestamp())

    count = df.count()
    df.write.format("delta").mode("overwrite").save(gold_path)
    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {count:,} category records | {elapsed}s")
    return {"table": "category_performance", "layer": "gold", "rows": count, "elapsed_s": elapsed}


def build_cohort_retention(spark, silver_base: str, gold_path: str) -> dict:
    t0 = time.time()
    print("\n[Gold] Building cohort_retention...")

    txns = spark.read.format("delta").load(f"{silver_base}/transactions")

    # First purchase month per customer
    first_purchase = txns.groupBy("customer_id").agg(
        F.date_format(F.min("order_date"), "yyyy-MM").alias("cohort_month")
    )

    df = txns.join(first_purchase, on="customer_id") \
             .withColumn("order_month", F.date_format("order_date", "yyyy-MM"))

    # Month index
    df = df.withColumn(
        "month_index",
        F.months_between(
            F.to_date(F.col("order_month"), "yyyy-MM"),
            F.to_date(F.col("cohort_month"), "yyyy-MM")
        ).cast("int")
    ).filter(F.col("month_index") >= 0)

    # Cohort sizes
    cohort_sizes = first_purchase.groupBy("cohort_month") \
                                  .agg(F.countDistinct("customer_id").alias("cohort_size"))

    # Active customers per cohort per month_index
    active = df.groupBy("cohort_month", "month_index") \
               .agg(F.countDistinct("customer_id").alias("active_customers"))

    result = active.join(cohort_sizes, on="cohort_month") \
                   .withColumn("retention_rate",
                               F.round(F.col("active_customers") / F.col("cohort_size") * 100, 1)) \
                   .filter(F.col("month_index") <= 11) \
                   .orderBy("cohort_month", "month_index")

    result = result.withColumn("_created_at", F.current_timestamp())

    count = result.count()
    result.write.format("delta").mode("overwrite").save(gold_path)
    elapsed = round(time.time() - t0, 2)
    print(f"  ✅ {count:,} cohort records | {elapsed}s")
    return {"table": "cohort_retention", "layer": "gold", "rows": count, "elapsed_s": elapsed}


def run_gold(delta_base: str = "delta") -> list:
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    s = f"{delta_base}/silver"
    g = f"{delta_base}/gold"
    os.makedirs(g, exist_ok=True)

    stats = []
    total_start = time.time()

    print("=" * 55)
    print("  GOLD LAYER — Business Aggregations")
    print("=" * 55)

    stats.append(build_customer_metrics(spark,     s, f"{g}/customer_metrics"))
    stats.append(build_monthly_revenue(spark,      s, f"{g}/monthly_revenue"))
    stats.append(build_category_performance(spark, s, f"{g}/category_performance"))
    stats.append(build_cohort_retention(spark,     s, f"{g}/cohort_retention"))

    total_time = round(time.time() - total_start, 2)
    print(f"\n{'=' * 55}")
    print(f"  Gold complete: {len(stats)} tables built in {total_time}s")
    print(f"{'=' * 55}")

    spark.stop()
    return stats


if __name__ == "__main__":
    stats = run_gold()
    for s in stats:
        print(f"  {s['table']:<25} {s['rows']:>8,} rows  {s['elapsed_s']}s")
