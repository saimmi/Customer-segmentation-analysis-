# Customer-segmentation-analysis-
# Customer Segmentation with RFM Analysis & KMeans Clustering

A complete, end-to-end data science project that takes a raw online retail transaction dataset and builds a customer segmentation system using RFM feature engineering and KMeans clustering. The final output is seven named customer groups, each with a clear business strategy.

---

## Overview

Most businesses send the same marketing message to all their customers — the loyal monthly buyer, the one-time visitor from a year ago, and the wholesale client spending thousands each month. This project replaces that one-size-fits-all approach with a data-driven segmentation system that groups customers by their actual purchasing behaviour.

**Business question answered:** Who are our customers, how are they different from each other, and what specific action should we take for each group?

---

## Dataset

- **Source:** [Online Retail II — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Period:** December 2009 – December 2010
- **Size:** 525,461 raw transactions, 8 columns
- **Scope:** UK-based e-commerce company, 40 countries, ~4,300 unique customers after cleaning

| Column | Description |
|---|---|
| `Invoice` | Unique order ID. Prefix `C` = cancellation, `A` = admin entry |
| `StockCode` | Product identifier |
| `Description` | Product name |
| `Quantity` | Units ordered (negative = return) |
| `InvoiceDate` | Timestamp of the order |
| `Price` | Unit price in GBP (£) |
| `Customer ID` | Unique customer number (107,927 rows missing) |
| `Country` | Customer's country |

---

## Project Structure

```
customer-segmentation/
│
├── data/
│   └── online_retail_II.xlsx          # Raw dataset (not included — see download link)
│
├── notebooks/
│   └── online-retail-data-clustering.ipynb   # Main analysis notebook
│
├── outputs/
│   └── customer_segmentation_case_study.html  # Full visual case study report
│
├── README.md
└── requirements.txt
```

---

## Methodology

### 1. Data Cleaning

About 23% of the raw data was removed before any analysis. Every removal was documented and deliberate — not arbitrary filtering.

| What was removed | Why | Rows affected |
|---|---|---|
| Invoices starting with `C` | Customer cancellations / returns (negative quantities) | ~10,209 |
| Invoices starting with `A` | Internal bad-debt write-offs (e.g. price of −£53,594) | 3 |
| Non-product stock codes (`DOT`, `M`, `BANK CHARGES`, `TEST001`, `AMAZONFEE`…) | Operational entries, not real product sales | ~2,456 |
| Rows with missing `Customer ID` | Anonymous transactions — cannot be assigned to a customer | ~107,927 |
| Rows with `Price == 0` | Free samples or data entry errors | 28 |

**Result:** 406,309 clean rows retained (77.3% of original).

---

### 2. Feature Engineering — RFM Model

Each customer's transaction history was collapsed into three metrics using `groupby` and aggregation.

```python
aggregated_df = cleaned_df.groupby("Customer ID", as_index=False).agg(
    MonetaryValue = ("SalesLineTotal", "sum"),
    Frequency     = ("Invoice",        "nunique"),
    LastInvoiceDate = ("InvoiceDate",  "max")
)

max_date = aggregated_df["LastInvoiceDate"].max()
aggregated_df["Recency"] = (max_date - aggregated_df["LastInvoiceDate"]).dt.days
```

| Feature | What it measures | Good value |
|---|---|---|
| **Recency** | Days since last purchase | Low (bought recently) |
| **Frequency** | Number of unique orders placed | High (loyal, returning customer) |
| **Monetary Value** | Total £ spent across all purchases | High (valuable customer) |

**Result:** 4,285 customers, each described by exactly 3 numbers.

---

### 3. Outlier Detection — IQR Method

Extreme outliers were separated *before* running KMeans. Including them distorts the cluster centres and makes all other segments less meaningful.

```python
# Monetary outliers
M_Q1, M_Q3 = df["MonetaryValue"].quantile([0.25, 0.75])
monetary_outliers = df[df["MonetaryValue"] > M_Q3 + 1.5 * (M_Q3 - M_Q1)]

# Frequency outliers
F_Q1, F_Q3 = df["Frequency"].quantile([0.25, 0.75])
frequency_outliers = df[df["Frequency"] > F_Q3 + 1.5 * (F_Q3 - F_Q1)]

# Non-outlier set — used for KMeans
non_outliers = df[
    ~df.index.isin(monetary_outliers.index) &
    ~df.index.isin(frequency_outliers.index)
]  # 3,809 customers
```

| Group | Count | Description |
|---|---|---|
| Regular customers (for KMeans) | 3,809 | Core customer base |
| Monetary outliers only | 197 | High spenders, avg £12,188 |
| Frequency outliers only | 53 | Very frequent buyers, avg 23 orders |
| Both outliers | 226 | Extreme on both dimensions |

---

### 4. Standard Scaling

Before clustering, all three features were scaled to have **mean = 0** and **standard deviation = 1**. This prevents Monetary Value (range: £1–£3,788) from dominating over Frequency (range: 1–11) in the distance calculations KMeans relies on.

```
z = (x − μ) / σ
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(non_outliers[["MonetaryValue", "Frequency", "Recency"]])
```

---

### 5. KMeans Clustering — Choosing k

The algorithm was run for k = 2 through k = 12. Two independent methods were used to select the optimal number of clusters.

**Elbow method (inertia):** Inertia drops sharply from k=2 to k=4, then the rate of decrease flattens out. The "elbow" is at k=4.

**Silhouette score:** Measures how well-separated the clusters are (range: −1 to +1). Score peaks at k=4.

```
s(i) = [ b(i) − a(i) ] / max( a(i), b(i) )

where:
  a(i) = avg distance to all other points in the same cluster
  b(i) = avg distance to all points in the nearest other cluster
```

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_scores, sil_scores = [], []

for k in range(2, 13):
    km = KMeans(n_clusters=k, random_state=42, max_iter=1000)
    labels = km.fit_predict(scaled_df)
    inertia_scores.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_df, labels))

# Both methods agree: k=4 is optimal

kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)
non_outliers["Cluster"] = kmeans.fit_predict(scaled_df)
```

---

## Results — The 7 Customer Segments

Clusters are numbered 0–3 by KMeans and −1 to −3 for outlier groups. Each was given a business-friendly name.

```python
cluster_labels = {
     0: "RETAIN",
     1: "RE-ENGAGE",
     2: "NURTURE",
     3: "REWARD",
    -1: "PAMPER",
    -2: "UPSELL",
    -3: "DELIGHT"
}
```

| Segment | Cluster | Count | Recency | Frequency | Monetary | Strategy |
|---|---|---|---|---|---|---|
| **RETAIN** | 0 | 1,120 | Medium | Medium | Medium-High | Loyalty programs, personalised offers |
| **RE-ENGAGE** | 1 | 980 | High (lapsed) | Low | Low | Win-back campaigns, time-limited discounts |
| **NURTURE** | 2 | 1,050 | Low (recent) | Low | Low | Welcome series, return-purchase incentives |
| **REWARD** | 3 | 659 | Low | High | High | VIP access, early releases, recognition |
| **PAMPER** | −1 | 262 | Varies | Low | Very High | Premium service, curated offers |
| **UPSELL** | −2 | 103 | Low | Very High | Low-Medium | Bundle deals, cross-sells, basket-size nudges |
| **DELIGHT** | −3 | 111 | Low | Very High | Extreme | Dedicated account manager, bespoke offers |

---

## Key Findings

**23% of raw data was unusable.** Cancellations, admin entries, zero-price rows, and anonymous transactions made up nearly 1 in 4 rows. Cleaning this properly is what makes the segments trustworthy.

**A small number of customers generate most of the revenue.** The DELIGHT segment is just 111 customers (~2.6% of the base) but likely accounts for a disproportionate share of total revenue. Losing one DELIGHT customer can matter more financially than losing dozens of RE-ENGAGE customers.

**Recency is the earliest warning sign of customer churn.** The RE-ENGAGE cluster is defined mainly by high recency — a long time since the last purchase. Tracking this metric in real time allows automated campaigns to trigger before a customer is fully lost.

**Separating outliers before clustering was a deliberate methodological choice.** Including DELIGHT and PAMPER customers in KMeans would have stretched all four clusters toward them, reducing the quality and actionability of the regular-customer segments.

**Both the elbow method and silhouette score independently confirmed k=4.** This removes guesswork from the most important decision in the clustering process.

---

## Libraries Used

```
pandas==2.x
numpy
matplotlib
seaborn
scikit-learn
openpyxl          # for reading .xlsx files
jupyter
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation

# 2. Download the dataset
# https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Place online_retail_II.xlsx inside the data/ folder

# 3. Install dependencies
pip install -r requirements.txt

# 4. Open the notebook
jupyter notebook notebooks/online-retail-data-clustering.ipynb
```

---

## Output Files

| File | Description |
|---|---|
| `notebooks/online-retail-data-clustering.ipynb` | Full analysis notebook with all code, plots, and markdown explanations |
| `outputs/customer_segmentation_case_study.html` | Standalone visual report with 12 interactive charts — open in any browser |

---

## Dataset Citation

Chen, D. (2015). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
