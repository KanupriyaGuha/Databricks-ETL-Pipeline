"""
app.py
Databricks ETL Pipeline Dashboard
Kanupriya Guha | Data Science Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETL Pipeline Dashboard | Kanupriya Guha",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS — Orange / Ember Theme ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d0806; color: #fef3c7; }
    section[data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"]  { display: none; }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    .block-container {
        max-width: 1200px !important;
        padding-top: 0.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: 0 auto !important;
    }

    /* Remove Streamlit's top header bar */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }

    h1, h2, h3 { color: #fef3c7 !important; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a0e06 0%, #150b04 100%);
        border: 1px solid #7c2d12;
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-value { font-size: 1.9rem; font-weight: 700; color: #fb923c; margin: 0; }
    .kpi-label { font-size: 0.78rem; color: #92400e; margin-top: 4px;
                 text-transform: uppercase; letter-spacing: 0.06em; color: #d97706; }
    .kpi-delta { font-size: 0.84rem; font-weight: 600; color: #fbbf24; margin-top: 5px; }

    /* Section headers */
    .section-header {
        font-size: 1.15rem; font-weight: 600; color: #fbbf24;
        border-left: 4px solid #f97316; padding-left: 12px;
        margin: 32px 0 14px 0;
    }

    /* Layer cards */
    .layer-bronze { background: linear-gradient(135deg, #1a0e06, #2d1206);
                    border: 1px solid #9a3412; border-radius: 14px; padding: 20px; text-align: center; }
    .layer-silver { background: linear-gradient(135deg, #0f1519, #0a1520);
                    border: 1px solid #334155; border-radius: 14px; padding: 20px; text-align: center; }
    .layer-gold   { background: linear-gradient(135deg, #1a1206, #201506);
                    border: 1px solid #92400e; border-radius: 14px; padding: 20px; text-align: center; }

    /* Tech pills */
    .pill {
        display: inline-block; background: #1c0f05; color: #fb923c;
        border: 1px solid #7c2d12; border-radius: 20px;
        padding: 4px 13px; font-size: 0.78rem; margin: 2px;
    }

    /* Badge styles */
    .badge-bronze  { background:#431407; color:#fb923c; border:1px solid #9a3412;
                     border-radius:6px; padding:2px 9px; font-size:0.78rem; font-weight:600; }
    .badge-silver  { background:#1e293b; color:#94a3b8; border:1px solid #475569;
                     border-radius:6px; padding:2px 9px; font-size:0.78rem; font-weight:600; }
    .badge-gold    { background:#422006; color:#fbbf24; border:1px solid #92400e;
                     border-radius:6px; padding:2px 9px; font-size:0.78rem; font-weight:600; }
    .badge-success { background:#052e16; color:#4ade80; border:1px solid #166534;
                     border-radius:6px; padding:2px 9px; font-size:0.78rem; font-weight:600; }
    .badge-fail    { background:#2d0a0a; color:#f87171; border:1px solid #991b1b;
                     border-radius:6px; padding:2px 9px; font-size:0.78rem; font-weight:600; }

    /* Run log row */
    .run-row {
        background: #120a04; border: 1px solid #1c110a; border-radius: 10px;
        padding: 12px 16px; margin-bottom: 8px;
        display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    }

    div[data-baseweb="tab-list"] { background: transparent; }
    div[data-baseweb="tab"]      { color: #78716c; }
    div[aria-selected="true"]    { color: #fb923c !important; border-bottom: 2px solid #f97316; }

    /* Dataframe overrides */
    .stDataFrame { border: 1px solid #7c2d12 !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
DELTA_BASE = "delta/gold"
ORANGE = "#f97316"
AMBER  = "#fbbf24"
RED    = "#f87171"
GREEN  = "#4ade80"
SLATE  = "#94a3b8"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1c110a"

def chart_layout(height=300, margin=None):
    m = margin or dict(l=20, r=20, t=20, b=40)
    return dict(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#d97706"),
        margin=m, height=height,
    )

@st.cache_data
def read_delta(path):
    """Read Gold Delta table — skip _delta_log subfolder."""
    parquet_files = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "_delta_log"]
        for f in files:
            if f.endswith(".parquet"):
                parquet_files.append(os.path.join(root, f))
    if not parquet_files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)


@st.cache_data
def load_gold():
    return (
        read_delta(f"{DELTA_BASE}/customer_metrics"),
        read_delta(f"{DELTA_BASE}/monthly_revenue"),
        read_delta(f"{DELTA_BASE}/category_performance"),
        read_delta(f"{DELTA_BASE}/cohort_retention"),
    )


@st.cache_data
def load_runs():
    if not os.path.exists("pipeline_runs.json"):
        return []
    with open("pipeline_runs.json") as f:
        return json.load(f)


customers, monthly, categories, cohort = load_gold()

# Deduplicate — raw parquet reads include files from all pipeline runs
# (Delta transaction log isn't consulted, so old run files pile up)
if not customers.empty:
    customers  = customers.drop_duplicates(subset=["customer_id"])
if not monthly.empty:
    monthly    = monthly.drop_duplicates(subset=["month"])
if not categories.empty:
    categories = categories.drop_duplicates(subset=["category"])
if not cohort.empty:
    cohort     = cohort.drop_duplicates(subset=["cohort_month", "month_index"])

runs   = load_runs()
latest = runs[-1] if runs else {}

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:4px 0 6px 0'>
    <h1 style='font-size:2.3rem; margin-bottom:6px; color:#fef3c7'>
        ⚡ Databricks ETL Pipeline
    </h1>
    <p style='color:#92400e; margin-top:0; font-size:0.95rem'>
        Medallion Architecture &nbsp;·&nbsp; Bronze → Silver → Gold &nbsp;·&nbsp;
        PySpark &nbsp;·&nbsp; Delta Lake &nbsp;·&nbsp; Data Quality Validation
    </p>
    <div style='margin:10px 0 4px 0'>
        <span class='pill'>PySpark</span>
        <span class='pill'>Delta Lake</span>
        <span class='pill'>Medallion Architecture</span>
        <span class='pill'>Data Validation</span>
        <span class='pill'>Python</span>
        <span class='pill'>Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2d1206; margin:8px 0 10px 0'>", unsafe_allow_html=True)

# ─── Pipeline KPIs ────────────────────────────────────────────────────────────
gen          = latest.get("data_generation", {})
stages       = latest.get("stages", [])
bronze_tables= stages[0].get("tables", []) if len(stages) > 0 else []
silver_tables= stages[1].get("tables", []) if len(stages) > 1 else []

total_raw    = gen.get("total_rows", 0)
total_clean  = sum(t.get("clean_rows", 0) for t in silver_tables)
total_dropped= sum(t.get("dropped_rows", 0) for t in silver_tables)
pipeline_time= latest.get("total_elapsed_s", 0)
run_status   = latest.get("status", "—").upper()

k1, k2, k3, k4, k5 = st.columns(5)
for col, val, label, delta in [
    (k1, f"{total_raw:,}",        "Raw Rows Ingested",    "4 source tables"),
    (k2, f"{total_clean:,}",      "Clean Rows (Silver)",  "After validation"),
    (k3, f"{total_dropped:,}",    "Bad Rows Removed",     "Quality filter"),
    (k4, f"{pipeline_time:.0f}s", "Pipeline Runtime",     "End-to-end"),
    (k5, run_status,              "Last Run Status",      latest.get("run_id","—")[-15:]),
]:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-delta'>{delta}</div>
        </div>""", unsafe_allow_html=True)


# ─── Medallion Architecture Cards ─────────────────────────────────────────────
st.markdown("<div class='section-header'>🏗️ Medallion Architecture — Layer Overview</div>",
            unsafe_allow_html=True)

a1, arr1, a2, arr2, a3 = st.columns([3, 0.4, 3, 0.4, 3])

bronze_rows = sum(t.get("rows", 0) for t in bronze_tables)
silver_rows = total_clean
gold_rows   = len(customers) + len(monthly) + len(categories) + len(cohort)
drop_pct    = round(total_dropped / bronze_rows * 100, 3) if bronze_rows > 0 else 0

with a1:
    st.markdown(f"""
    <div class='layer-bronze'>
        <div style='font-size:2rem'>🥉</div>
        <div style='font-size:1.6rem; font-weight:700; color:#fb923c; margin:4px 0'>{bronze_rows:,}</div>
        <div style='color:#d97706; font-weight:600; font-size:0.95rem'>BRONZE LAYER</div>
        <div style='color:#92400e; font-size:0.80rem; margin-top:6px'>Raw ingestion · No transforms<br>4 Delta tables · Schema enforced</div>
    </div>""", unsafe_allow_html=True)

with arr1:
    st.markdown("<div style='text-align:center; padding-top:55px; font-size:1.8rem; color:#7c2d12'>→</div>",
                unsafe_allow_html=True)

with a2:
    st.markdown(f"""
    <div class='layer-silver'>
        <div style='font-size:2rem'>🥈</div>
        <div style='font-size:1.6rem; font-weight:700; color:#94a3b8; margin:4px 0'>{silver_rows:,}</div>
        <div style='color:#64748b; font-weight:600; font-size:0.95rem'>SILVER LAYER</div>
        <div style='color:#475569; font-size:0.80rem; margin-top:6px'>Cleaned & validated<br>{drop_pct}% bad rows removed · Ref. integrity</div>
    </div>""", unsafe_allow_html=True)

with arr2:
    st.markdown("<div style='text-align:center; padding-top:55px; font-size:1.8rem; color:#7c2d12'>→</div>",
                unsafe_allow_html=True)

with a3:
    st.markdown(f"""
    <div class='layer-gold'>
        <div style='font-size:2rem'>🥇</div>
        <div style='font-size:1.6rem; font-weight:700; color:#fbbf24; margin:4px 0'>{gold_rows:,}</div>
        <div style='color:#d97706; font-weight:600; font-size:0.95rem'>GOLD LAYER</div>
        <div style='color:#92400e; font-size:0.80rem; margin-top:6px'>Business aggregations<br>4 analytical tables · Query-ready</div>
    </div>""", unsafe_allow_html=True)


# ─── Pipeline Stage Timing ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⏱️ Pipeline Stage Timing</div>",
            unsafe_allow_html=True)

if stages:
    timing_rows = []
    for stage in stages:
        layer = stage.get("layer", "")
        for t in stage.get("tables", []):
            timing_rows.append({
                "label":   f"{layer.upper()} · {t.get('table','')}",
                "elapsed": t.get("elapsed_s", 0),
                "layer":   layer,
            })
    timing_df = pd.DataFrame(timing_rows)

    color_map = {"bronze": "#fb923c", "silver": "#94a3b8", "gold": "#fbbf24"}
    colors = [color_map.get(r["layer"], ORANGE) for _, r in timing_df.iterrows()]

    fig_timing = go.Figure(go.Bar(
        y=timing_df["label"],
        x=timing_df["elapsed"],
        orientation="h",
        marker_color=colors,
        text=[f"{v}s" for v in timing_df["elapsed"]],
        textposition="outside",
        textfont=dict(color="#fef3c7", size=11),
    ))
    fig_timing.update_layout(
        **chart_layout(height=320, margin=dict(l=10, r=60, t=10, b=30)),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706"), title="Seconds"),
        yaxis=dict(tickfont=dict(color="#fef3c7")),
    )
    st.plotly_chart(fig_timing, use_container_width=True)


# ─── Data Quality ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔍 Data Quality — Silver Layer Validation</div>",
            unsafe_allow_html=True)

if silver_tables:
    dq_l, dq_r = st.columns([2, 1])

    with dq_l:
        dq_df = pd.DataFrame([{
            "Table":      t.get("table",""),
            "Clean Rows": t.get("clean_rows", 0),
            "Dropped":    t.get("dropped_rows", 0),
            "Drop %":     t.get("drop_pct", 0),
        } for t in silver_tables])

        fig_dq = go.Figure()
        fig_dq.add_trace(go.Bar(
            name="Clean",   x=dq_df["Table"], y=dq_df["Clean Rows"],
            marker_color=ORANGE,
            text=[f"{v:,}" for v in dq_df["Clean Rows"]],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#0d0806", size=11, family="monospace"),
        ))
        fig_dq.add_trace(go.Bar(
            name="Dropped", x=dq_df["Table"], y=dq_df["Dropped"],
            marker_color=RED,
            text=[f"{v:,}" if v > 0 else "" for v in dq_df["Dropped"]],
            textposition="outside",
            textfont=dict(color=RED, size=11),
        ))
        fig_dq.update_layout(
            barmode="group",
            **chart_layout(height=280, margin=dict(l=10, r=10, t=10, b=40)),
            xaxis=dict(tickfont=dict(color="#fef3c7"), gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706"), type="log"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#fef3c7")),
        )
        st.plotly_chart(fig_dq, use_container_width=True)
        st.caption("Y-axis is log scale — makes small tables (products: 500 rows) visible alongside large ones (transactions: 1M rows)")

    with dq_r:
        st.markdown("<br>", unsafe_allow_html=True)
        for t in silver_tables:
            quality_score = 100 - t.get("drop_pct", 0)
            color = "#4ade80" if quality_score > 99 else ("#fbbf24" if quality_score > 95 else RED)
            st.markdown(f"""
            <div style='background:#120a04; border:1px solid #2d1206; border-radius:10px;
                        padding:12px 14px; margin-bottom:8px'>
                <div style='color:#fef3c7; font-weight:600; font-size:0.90rem'>{t.get("table","").upper()}</div>
                <div style='color:{color}; font-size:1.3rem; font-weight:700'>{quality_score:.1f}%</div>
                <div style='color:#78716c; font-size:0.75rem'>quality score</div>
                <div style='color:#d97706; font-size:0.78rem; margin-top:4px'>
                    {t.get("clean_rows",0):,} clean &nbsp;·&nbsp; {t.get("dropped_rows",0):,} dropped
                </div>
            </div>""", unsafe_allow_html=True)


# ─── Revenue Trends ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Revenue Trends (Gold Layer)</div>",
            unsafe_allow_html=True)

if not monthly.empty:
    monthly = monthly.sort_values("month")
    rev_tab, aov_tab, orders_tab, returns_tab = st.tabs([
        "💰 Monthly Revenue", "🧾 Avg Order Value", "📦 Order Volume", "↩️ Return Rate"
    ])

    for tab, y_col, color, prefix, name in [
        (rev_tab,     "net_revenue",      ORANGE, "$", "Net Revenue"),
        (aov_tab,     "avg_order_value",  AMBER,  "$", "Avg Order Value"),
        (orders_tab,  "total_orders",     GREEN,  "",  "Orders"),
        (returns_tab, "return_rate",      RED,    "",  "Return Rate %"),
    ]:
        with tab:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly["month"], y=monthly[y_col],
                mode="lines+markers", name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=6, color=color),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            ))
            fig.update_layout(
                **chart_layout(height=270, margin=dict(l=40, r=20, t=10, b=80)),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706"), tickangle=45),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706")),
            )
            st.plotly_chart(fig, use_container_width=True)


# ─── Category Performance ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🛍️ Category Performance (Gold Layer)</div>",
            unsafe_allow_html=True)

if not categories.empty:
    cat_l, cat_r = st.columns(2)

    with cat_l:
        fig_rev = go.Figure(go.Bar(
            x=categories["net_revenue"],
            y=categories["category"],
            orientation="h",
            marker=dict(
                color=categories["net_revenue"],
                colorscale=[[0, "#7c2d12"], [0.5, "#ea580c"], [1, "#fbbf24"]],
                showscale=False,
            ),
            text=categories["net_revenue"].apply(lambda x: f"${x/1e6:.1f}M"),
            textposition="outside",
            textfont=dict(color="#fef3c7", size=11),
        ))
        fig_rev.update_layout(
            title=dict(text="Net Revenue by Category", font=dict(color="#fbbf24", size=13)),
            **chart_layout(height=360, margin=dict(l=10, r=90, t=40, b=10)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706")),
            yaxis=dict(tickfont=dict(color="#fef3c7")),
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with cat_r:
        fig_ret = go.Figure(go.Bar(
            x=categories["return_rate"],
            y=categories["category"],
            orientation="h",
            marker=dict(
                color=categories["return_rate"],
                colorscale=[[0, "#052e16"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=False,
            ),
            text=categories["return_rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            textfont=dict(color="#fef3c7", size=11),
        ))
        fig_ret.update_layout(
            title=dict(text="Return Rate by Category", font=dict(color="#fbbf24", size=13)),
            **chart_layout(height=360, margin=dict(l=10, r=60, t=40, b=10)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706")),
            yaxis=dict(tickfont=dict(color="#fef3c7")),
        )
        st.plotly_chart(fig_ret, use_container_width=True)


# ─── Customer Metrics ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>👥 Customer Metrics (Gold Layer)</div>",
            unsafe_allow_html=True)

if not customers.empty:
    cm1, cm2, cm3, cm4 = st.columns(4)
    for col, val, label, delta in [
        (cm1, f"{len(customers):,}",                          "Total Customers",      "All segments"),
        (cm2, f"${customers['net_revenue'].mean():,.0f}",     "Avg Net LTV",          "Per customer"),
        (cm3, f"{customers['total_orders'].mean():.1f}",      "Avg Orders",           "Per customer"),
        (cm4, f"{(customers['churn_risk']=='High').sum():,}", "High Churn Risk",       "Inactive >180d"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='padding:14px 12px'>
                <div class='kpi-value' style='font-size:1.55rem'>{val}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-delta'>{delta}</div>
            </div>""", unsafe_allow_html=True)

    cust_l, cust_r = st.columns(2)

    with cust_l:
        churn_counts = customers["churn_risk"].value_counts().reset_index()
        churn_counts.columns = ["risk", "count"]
        cmap = {"Low": GREEN, "Medium": AMBER, "High": RED}
        fig_c = go.Figure(go.Pie(
            labels=churn_counts["risk"],
            values=churn_counts["count"],
            marker_colors=[cmap.get(r, SLATE) for r in churn_counts["risk"]],
            hole=0.52,
            textinfo="label+percent",
            textfont=dict(color="#fef3c7", size=12),
        ))
        fig_c.update_layout(
            title=dict(text="Churn Risk Distribution", font=dict(color="#fbbf24", size=13)),
            paper_bgcolor=CHART_BG, font=dict(color="#d97706"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#fef3c7")),
            margin=dict(l=10, r=10, t=40, b=10), height=280,
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with cust_r:
        tier_rev = customers.groupby("tier")["net_revenue"].mean().reset_index()
        tier_color = {"Platinum": "#e879f9", "Gold": AMBER, "Silver": SLATE, "Bronze": ORANGE}
        fig_t = go.Figure(go.Bar(
            x=tier_rev["tier"], y=tier_rev["net_revenue"],
            marker_color=[tier_color.get(t, ORANGE) for t in tier_rev["tier"]],
            text=tier_rev["net_revenue"].apply(lambda x: f"${x:,.0f}"),
            textposition="outside", textfont=dict(color="#fef3c7", size=11),
        ))
        fig_t.update_layout(
            title=dict(text="Avg Revenue by Customer Tier", font=dict(color="#fbbf24", size=13)),
            **chart_layout(height=280, margin=dict(l=10, r=10, t=40, b=20)),
            xaxis=dict(tickfont=dict(color="#fef3c7")),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color="#d97706")),
        )
        st.plotly_chart(fig_t, use_container_width=True)


# ─── Top 10 Customers ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🏆 Top 10 Customers by Lifetime Value (Gold Layer)</div>",
            unsafe_allow_html=True)

if not customers.empty:
    top10 = customers.nlargest(10, "net_revenue")[[
        "customer_id", "tier", "net_revenue", "total_orders",
        "total_returns", "recency_days", "churn_risk"
    ]].copy()
    top10.columns = ["Customer ID", "Tier", "Net Revenue ($)", "Orders", "Returns", "Recency (days)", "Churn Risk"]
    top10["Net Revenue ($)"] = top10["Net Revenue ($)"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(top10, use_container_width=True, hide_index=True, height=420)


# ─── Cohort Retention Heatmap ─────────────────────────────────────────────────
st.markdown("<div class='section-header'>📅 Cohort Retention Analysis (Gold Layer)</div>",
            unsafe_allow_html=True)

if not cohort.empty:
    cohort_pivot = cohort.pivot_table(
        index="cohort_month", columns="month_index",
        values="retention_rate", aggfunc="mean"
    )
    cohort_pivot = cohort_pivot[[c for c in sorted(cohort_pivot.columns) if c <= 11]]
    cohort_pivot.columns = [f"M+{int(c)}" for c in cohort_pivot.columns]
    cohort_pivot.index = cohort_pivot.index.astype(str)
    cohort_pivot = cohort_pivot.sort_index()

    text_vals = [
        [f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
        for row in cohort_pivot.values
    ]

    fig_hm = go.Figure(go.Heatmap(
        z=cohort_pivot.values.tolist(),
        x=cohort_pivot.columns.tolist(),
        y=cohort_pivot.index.tolist(),
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        colorscale=[
            [0.0, "#0d0806"],
            [0.2, "#431407"],
            [0.4, "#9a3412"],
            [0.6, "#ea580c"],
            [0.8, "#fb923c"],
            [1.0, "#fef3c7"],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Retention %", font=dict(color="#d97706", size=11)),
            tickfont=dict(color="#d97706"),
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
        ),
    ))
    n_cohorts = len(cohort_pivot)
    fig_hm.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        font=dict(color="#d97706"),
        xaxis=dict(tickfont=dict(color="#d97706"), title="Months Since First Purchase"),
        yaxis=dict(tickfont=dict(color="#fef3c7"), title="Cohort Month", autorange="reversed"),
        margin=dict(l=20, r=20, t=10, b=60),
        height=max(400, n_cohorts * 18),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("M+0 = first purchase month (100%). Each row shows % of that cohort who returned each subsequent month.")


# ─── Pipeline Run History ─────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Pipeline Run History</div>",
            unsafe_allow_html=True)

if runs:
    for run in reversed(runs[-5:]):
        badge = ("<span class='badge-success'>✅ SUCCESS</span>"
                 if run.get("status") == "success"
                 else "<span class='badge-fail'>❌ FAILED</span>")
        started = run.get("started_at","")[:19].replace("T"," ")
        elapsed = run.get("total_elapsed_s","—")
        rows    = run.get("data_generation",{}).get("total_rows", 0)
        st.markdown(f"""
        <div class='run-row'>
            <span style='color:#fb923c; font-weight:600; font-family:monospace; font-size:0.88rem'>
                {run.get("run_id","")}
            </span>
            {badge}
            <span style='color:#78716c; font-size:0.84rem'>🕐 {started}</span>
            <span style='color:#78716c; font-size:0.84rem'>⏱ {elapsed}s</span>
            <span style='color:#78716c; font-size:0.84rem'>📦 {rows:,} rows</span>
            <span class='badge-bronze'>Bronze</span>
            <span class='badge-silver'>Silver</span>
            <span class='badge-gold'>Gold</span>
        </div>""", unsafe_allow_html=True)
else:
    st.info("No pipeline runs found. Run pipeline.py first.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2d1206; margin-top:40px'>
<div style='text-align:center; color:#78716c; font-size:0.80rem; padding-bottom:20px'>
    Built by <b style='color:#fb923c'>Kanupriya Guha</b> ·
    Databricks ETL Pipeline · Data Engineering Portfolio ·
    <a href='https://github.com/kanupriyaguha' style='color:#f97316'>GitHub</a> ·
    <a href='https://linkedin.com/in/kanupriyaguha' style='color:#f97316'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
