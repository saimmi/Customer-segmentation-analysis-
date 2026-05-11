
# Customer-segmentation-analysis-

# Customer Segmentation with RFM Analysis & KMeans Clustering

A complete, end-to-end data science project that takes a raw online retail transaction dataset and builds a customer segmentation system using RFM feature engineering and KMeans clustering. The final output is seven named customer groups, each with a clear business strategy.

---

## Project Highlights

- Cleaned and transformed **525K+ retail transactions**
- Built a full **RFM-based customer analytics pipeline**
- Applied **KMeans clustering** for behavioral segmentation
- Used **Elbow Method** and **Silhouette Score** to determine optimal clusters
- Identified **7 actionable customer groups**
- Converted technical clustering output into **business-ready strategies**
- Performed complete **EDA, feature engineering, scaling, outlier analysis, and modeling**
- Created a structured, reproducible **data science workflow**

---

## Business Problem

Most companies market to every customer in the same way, even though customer behaviour differs dramatically.

This project answers:

- Which customers are most valuable?
- Which customers are likely to churn?
- Which customers should receive loyalty rewards?
- Which customers need re-engagement campaigns?
- How can businesses increase retention and revenue using segmentation?

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

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |
| Notebook Environment | Jupyter Notebook |
| File Handling | openpyxl |

---

## Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Customer Analytics
- RFM Analysis
- Outlier Detection
- Data Standardization
- Unsupervised Machine Learning
- KMeans Clustering
- Cluster Evaluation
- Business Intelligence
- Business Strategy Translation
- Data Visualization
- Analytical Thinking

---

## Project Structure

```bash
customer-segmentation/
│
├── data/
│   └── online_retail_II.xlsx
│
├── notebooks/
│   └── online-retail-data-clustering.ipynb
│
├── outputs/
│   └── customer_segmentation_case_study.html
│
├── images/
│   ├── elbow_method.png
│   ├── silhouette_score.png
│   ├── cluster_distribution.png
│   └── rfm_heatmap.png
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
| Invoices starting with `A` | Internal bad-debt write-offs | 3 |
| Non-product stock codes | Operational entries, not real sales | ~2,456 |
| Missing `Customer ID` | Cannot assign transactions to customers | ~107,927 |
| Zero-price rows | Free samples / errors | 28 |

**Result:** 406,309 clean rows retained (77.3% of original).

---

### 2. Feature Engineering — RFM Model

```python
aggregated_df = cleaned_df.groupby("Customer ID", as_index=False).agg(
    MonetaryValue = ("SalesLineTotal", "sum"),
    Frequency     = ("Invoice", "nunique"),
    LastInvoiceDate = ("InvoiceDate", "max")
)
```

| Feature | Meaning |
|---|---|
| Recency | Days since last purchase |
| Frequency | Number of purchases |
| Monetary Value | Total amount spent |

---

### 3. Outlier Detection

Used the **IQR method** to isolate extreme customers before clustering.

Why?

Because high-spending enterprise buyers can distort cluster centers and reduce segmentation quality.

---

### 4. Feature Scaling

Applied `StandardScaler()` so all variables contribute equally to distance calculations.

```python
from sklearn.preprocessing import StandardScaler
```

---

### 5. KMeans Clustering

Used:

- Elbow Method
- Silhouette Score

Both methods confirmed:

```python
k = 4
```

---

## Results — The 7 Customer Segments

| Segment | Behaviour | Business Action |
|---|---|---|
| RETAIN | Stable repeat customers | Loyalty programs |
| RE-ENGAGE | Inactive customers | Win-back campaigns |
| NURTURE | New / low-frequency buyers | Onboarding offers |
| REWARD | Loyal high-value customers | VIP rewards |
| PAMPER | Big spenders | Premium treatment |
| UPSELL | Frequent but low-spend | Bundle recommendations |
| DELIGHT | Elite customers | Dedicated relationship management |

---

## Key Business Insights

### 1. Small customer groups generate disproportionate revenue

The DELIGHT segment represents a tiny percentage of customers but likely contributes a major share of total sales.

### 2. Recency is a powerful churn signal

Customers with increasing recency are at high risk of disengagement.

### 3. Outlier handling improved segmentation quality

Separating extreme spenders before clustering produced cleaner and more interpretable customer groups.

### 4. Customer segmentation enables personalized marketing

Different customer types require different retention and engagement strategies.

---

## Visualizations Included

- Distribution plots
- Correlation heatmaps
- Boxplots for outlier detection
- Elbow curve
- Silhouette score analysis
- Cluster distribution charts
- RFM comparison plots
- Cluster behavior heatmaps

---

## Libraries Used

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
jupyter
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
# Clone repository
git clone https://github.com/yourusername/customer-segmentation.git

# Enter project folder
cd customer-segmentation

# Install requirements
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## Future Improvements

- Deploy segmentation dashboard using Streamlit or Power BI
- Add automated customer scoring pipeline
- Experiment with DBSCAN and Hierarchical Clustering
- Build churn prediction models
- Add recommendation systems
- Perform time-series customer behaviour analysis

---

## Output Files

| File | Description |
|---|---|
| `online-retail-data-clustering.ipynb` | Full notebook with analysis |
| `customer_segmentation_case_study.html` | Interactive business report |
| `requirements.txt` | Project dependencies |

---

## Sample Business Recommendations

| Segment | Recommendation |
|---|---|
| RE-ENGAGE | Send limited-time offers |
| REWARD | Give VIP membership |
| NURTURE | Educate with onboarding emails |
| UPSELL | Cross-sell complementary products |
| DELIGHT | Offer premium concierge support |

---

## Learning Outcomes

Through this project, I gained hands-on experience in:

- Real-world data cleaning
- Feature engineering
- Customer analytics
- Machine learning workflows
- Cluster evaluation techniques
- Translating technical analysis into business insights
- End-to-end project structuring for production-ready portfolios

---

## Author

### Saimmi Nisha

Aspiring Data Analyst passionate about:

- Data Analytics
- Business Intelligence
- Machine Learning
- Customer Insights
- Data Visualization

---

## Connect With Me

- LinkedIn
- GitHub
- Portfolio

(Add your links here)

---

## Dataset Citation

Chen, D. (2015). *Online Retail II* [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C5CG6D
````
