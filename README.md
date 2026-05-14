# ⚡ Databricks ETL Pipeline — Medallion Architecture

A production-style ETL pipeline built with PySpark and Delta Lake, processing 1.18M+ records across 4 tables through Bronze → Silver → Gold layers with automated data quality validation.

**Live Demo → [databricks-etl-pipeline.streamlit.app](https://databricks-etl-pipeline.streamlit.app)**

---

## Architecture

```
Raw CSVs  →  🥉 Bronze  →  🥈 Silver  →  🥇 Gold  →  📊 Dashboard
              (Ingest)     (Validate)   (Aggregate)
```

### Bronze Layer — Raw Ingestion
- Loads 4 CSV sources into Delta tables as-is
- Enforces schema (data types, nullability)
- Adds ingestion metadata (`_ingested_at`, `_source_file`)
- No business logic — pure landing zone

### Silver Layer — Clean & Validate
- Removes duplicates, null primary keys, invalid values
- Fills missing fields with sensible defaults
- **Referential integrity checks** — transactions must reference valid customers and products
- Flags and removes bad records (negative amounts, invalid dates, orphaned references)
- Result: **2,990 bad rows removed** (99.75% data quality score on transactions)

### Gold Layer — Business Aggregations
Produces 4 query-ready analytical tables:

| Table | Description |
|-------|-------------|
| `customer_metrics` | One row per customer — LTV, churn risk, tier, recency |
| `monthly_revenue` | Monthly revenue, AOV, order volume, return rate |
| `category_performance` | Revenue and return rate by product category |
| `cohort_retention` | Month-over-month retention rates by acquisition cohort |

---

## Dataset

Synthetic e-commerce data generated at scale — designed to reflect realistic buying behavior:

| Table | Rows | Description |
|-------|------|-------------|
| customers.csv | 100,000 | Demographics, signup date, tier, region |
| products.csv | 500 | 10 categories, price ranges, active status |
| transactions.csv | 1,000,000 | 3 years of orders (Jan 2022 – Dec 2024) |
| returns.csv | 79,760 | 8% return rate, linked to transactions |
| **Total** | **1,180,260** | |

Intentional dirty data included — ~0.3% negative amounts, ~0.5% null payment methods, ~2% missing emails, ~1% missing regions — so Silver validation has real work to do.

---

## Dashboard Features

- **Pipeline KPIs** — raw rows ingested, clean rows, bad rows removed, runtime, status
- **Medallion flow diagram** — Bronze → Silver → Gold with live row counts
- **Pipeline stage timing** — per-table execution time across all 3 layers
- **Data quality report** — log-scale bar chart + quality score per table
- **Revenue trends** — monthly revenue, AOV, order volume, return rate (tabbed)
- **Category performance** — net revenue and return rate by product category
- **Customer metrics** — churn risk distribution, avg revenue by tier
- **Top 10 customers** — highest LTV customers with orders, returns, recency
- **Cohort retention heatmap** — 36-month acquisition cohort analysis
- **Pipeline run history** — timestamped log of all pipeline executions

---

## Tech Stack

| Tool | Role |
|------|------|
| PySpark 3.5 | Distributed data processing |
| Delta Lake 3.1 | ACID transactions, versioned storage |
| Python | Orchestration & data generation |
| Pandas + PyArrow | Dashboard data reads |
| Plotly | Interactive charts |
| Streamlit | Dashboard framework |

---

## Project Structure

```
databricks_etl/
├── generate_data.py   # Synthetic data generator (1.18M rows, 4 tables)
├── bronze.py          # Bronze layer — raw CSV → Delta ingestion
├── silver.py          # Silver layer — cleaning, validation, ref. integrity
├── gold.py            # Gold layer — business aggregations
├── pipeline.py        # Orchestrator — Bronze → Silver → Gold + run logging
├── app.py             # Streamlit dashboard
├── requirements.txt
└── delta/gold/        # Pre-built Gold Delta tables (dashboard reads these)
```

---

## Run Locally

**Prerequisites:** Python 3.9+, Java 11+ (required for PySpark)

```bash
git clone https://github.com/KanupriyaGuha/Databricks-ETL-Pipeline.git
cd Databricks-ETL-Pipeline

pip install pyspark==3.5.0 delta-spark==3.1.0 streamlit pandas numpy plotly pyarrow

# Run the full pipeline (generates data + Bronze + Silver + Gold)
python pipeline.py

# Launch the dashboard
streamlit run app.py
```

---

## Resume Line

> *Built end-to-end PySpark ETL pipeline on Delta Lake medallion architecture (Bronze/Silver/Gold), processing 1.18M+ records across 4 tables with automated data quality validation, referential integrity checks, and business-layer aggregations*

---

Built by **Kanupriya Guha** · [LinkedIn](https://linkedin.com/in/kanupriyaguha) · [GitHub](https://github.com/kanupriyaguha)
