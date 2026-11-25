# UrbanCart Customer Segmentation

---

## 1. The Problem

UrbanCart's marketing team was running "one-size-fits-all" campaigns. They suspected their customer base contained genuinely different shopper types, but they had:

- **No predefined categories** to sort customers into, and
- **No way of knowing how many groups actually exist.**

Since the segments are unlabeled, this is an **unsupervised learning** problem. The goal is to let the data reveal its natural structure, then translate each discovered cluster into a clear business profile so UrbanCart can replace blanket messaging with precision-targeted, automated campaigns.

### Dataset

The dataset consists of 20,000 customers, 4 behavioral features, no missing values, no duplicates.

| Feature | Description |
|---|---|
| `AvgOrderValue` | Average amount spent per order |
| `PurchaseFrequency` | Number of purchases per quarter |
| `NumProductCategoriesShoppedIn` | Number of distinct product categories bought from |
| `AvgDiscountUsed` | Average share of discounts/coupons applied to purchases |

---

## 2. The Approach

### Exploratory data analysis
- **Distributions:** `AvgOrderValue`, `PurchaseFrequency` and `AvgDiscountUsed` are all **bimodal** (two distinct peaks), not simply skewed. `NumProductCategoriesShoppedIn` is multimodal, with most customers shopping for products from two different categories.
- **Pairwise plots:** a clean two-group split driven by order value. One group has low order value, high frequency and heavy discount use. The other has high order value, low frequency and little discount use. There is a visible gap in `AvgOrderValue` between roughly $90 and $100.
- **Correlation:** `PurchaseFrequency` and `NumProductCategoriesShoppedIn` are moderately correlated (~0.6). I kept both features because the correlation is not severe, the scatter plots show each adds information the other lacks, and with only four features, dropping one would lose more information than it saves.

### Preprocessing
- Checked for duplicates and missing values (none found).
- Applied **`StandardScaler`** to all features, because K-Means relies on Euclidean distance and features with larger ranges (like order value) would otherwise dominate.
- **Deliberately did not log-transform.** The features look skewed but are really bimodal. A log transform would compress the axis and risk blurring the very gap the model needs to detect.

### Modeling
- **Algorithm: K-Means** (`init="k-means++"`, `random_state=1`). Chosen over hierarchical clustering because EDA already suggested a small number of clusters, K-Means lets us search a range of K with evaluation metrics, and it scales linearly to 20,000 rows. Agglomerative clustering would need a full pairwise distance matrix, which is costly at this size.
- **Choosing K (2 to 10):** used both the **Elbow method** (WCSS) and the **Silhouette score**. Both pointed to **K = 4**, which had the highest silhouette score (0.599).
- **Visualization:** since the data has four dimensions, PCA reduced it to 2D for plotting. The two components capture about 75% of the variance.
- **Interpretation:** cluster labels were mapped back onto the original (unscaled) data, and the mean of each feature per cluster was compared against the overall average to build the personas.

### Tech stack
Python, pandas, NumPy, scikit-learn (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`), Matplotlib, Seaborn, joblib.

---

## 3. The Results

### Choosing K

| K | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | 0.362 | 0.503 | **0.599** | 0.564 | 0.581 | 0.554 | 0.514 | 0.542 | 0.504 |

K = 4 gives the best-separated clusters. A silhouette score near 0.6 indicates well-defined, reasonably distinct groups.

### The four customer segments

| Cluster | Persona | Customers | Avg Order Value | Purchase Freq. | Categories | Avg Discount |
|---|---|---|---|---|---|---|
| 0 | **Premium Spenders** | 5,898 (29.5%) | $199.81 | 2.01 | 1.99 | 5% |
| 1 | **At-Risk Shoppers** | 3,242 (16.2%) | $26.38 | 1.31 | 1.12 | 3% |
| 2 | **Loyal Enthusiasts** | 5,803 (29.0%) | $35.15 | 12.07 | 3.03 | 10% |
| 3 | **Deal Seekers** | 5,057 (25.3%) | $44.97 | 3.00 | 2.01 | 35% |
| | *Overall average* | 20,000 | $84.77 | 5.06 | 2.16 | 14% |

### What each segment means and what to do about it

- **Premium Spenders:** More than double the average order value, but they shop rarely and almost never use discounts. They buy on quality or need, not price.
  **Action:** stop sending them coupons (it only burns margin). Target them with premium lines, early-access collections and high-touch service.

- **Loyal Enthusiasts:** The most frequent buyers (about 12 purchases a quarter, versus a 5 average) and the broadest cross-shoppers (3+ categories). Baskets are small, but the steady repeat business makes them the revenue backbone.
  **Action:** personalized cross-category recommendations and a structured loyalty rewards program.

- **Deal Seekers:** Near-average spend and frequency, but discount usage is about 2.5x the baseline. They are motivated by promotions.
  **Action:** use them to clear inventory through flash sales and end-of-season clearances, and use volume-based incentives (e.g. "spend $60 to unlock 35% off") to lift basket size.

- **At-Risk Shoppers:** Below average on every metric, shopping in about one category and about once a quarter. They look like one-and-done buyers on the verge of churning.
  **Action:** low-cost automated win-back email sequences with a strong "come back" incentive, rather than heavy manual ad spend.

### Key takeaways
- The customer base is not one audience. Order value, frequency and discount sensitivity separate customers into four clearly different groups.
- Two of the four segments (Premium Spenders and Loyal Enthusiasts) account for about 58% of customers and are the main targets for margin and retention-focused strategies.
- Discount-driven spend is concentrated in one segment, so blanket coupons are wasted on the other three.

### Limitations and next steps
- PCA plots show the 2D view captures ~75% of variance, so some detail is lost in the visualization.
- The silhouette score for K = 6 (0.581) is close to K = 4, so a finer segmentation could be explored if the business needs more granular groups.
- Next steps: validate clusters against actual campaign response (A/B tests), monitor for segment drift as new data arrives, and wire the saved model into an automated campaign pipeline.

---

## Repository contents

- `UrbanCart_Customer_Segmentation.ipynb`: full analysis (EDA, preprocessing, modeling, interpretation)
- `segment_model.pkl`: saved trained K-Means model (K = 4). Note it expects **standardized** inputs, so reuse the same fitted `StandardScaler` when scoring new customers.
- `README.md`: This file
