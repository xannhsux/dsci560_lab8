# Lab 8: Doc2Vec Embedding Configuration Analysis
## Comparing Three Doc2Vec Configurations for Reddit Post Clustering

**Course:** DSCI 560
**Task:** Compare three different Doc2Vec configurations for clustering Reddit posts
**Dataset:** 5,000 posts from r/iphone subreddit
**Date:** October 19, 2025

---

## Executive Summary

This analysis evaluated three different Doc2Vec embedding configurations to determine which produces the best clustering results for Reddit posts. Using both quantitative metrics and qualitative cluster analysis, **Configuration 1 (vector_size=50, min_count=1, epochs=40)** emerged as the recommended configuration with a composite score of 0.5157.

### Key Findings:
- **Best Configuration:** vector_size=50, min_count=1, epochs=40
- **Clusters Produced:** 4 well-defined clusters
- **Best Silhouette Score:** 0.0987 (vs200 configuration)
- **Best Cluster Separation:** vs50 configuration (lowest Davies-Bouldin Index)
- **Best Cluster Definition:** vs50 configuration (highest Calinski-Harabasz Score)

---

## 1. Methodology

### 1.1 Data Collection
- **Source:** r/iphone subreddit
- **Total Posts:** 5,000 unique posts
- **Time Range:** September 7, 2016 to October 19, 2025
- **Processing:** All posts were cleaned, tokenized, and preprocessed

### 1.2 Doc2Vec Configurations Tested

| Configuration | Vector Size | Min Count | Epochs | Description |
|--------------|-------------|-----------|--------|-------------|
| **Config 1** | 50 | 1 | 40 | Smaller embedding space, all words included |
| **Config 2** | 100 | 2 | 50 | Medium embedding, rare words filtered |
| **Config 3** | 200 | 1 | 60 | Large embedding space, all words included |

**Rationale for selections:**
- **Vector Size:** Tested small (50), medium (100), and large (200) to evaluate dimensionality impact
- **Min Count:** Tested inclusive (1) vs. filtered (2) to assess rare word impact
- **Epochs:** Increased training iterations with larger vector sizes

### 1.3 Evaluation Metrics

#### Quantitative Metrics:

1. **Silhouette Score** (range: -1 to 1, higher is better)
   - Measures how similar an object is to its own cluster compared to other clusters
   - > 0.7: Strong structure
   - 0.5-0.7: Reasonable structure
   - 0.25-0.5: Weak structure
   - < 0.25: No substantial structure

2. **Davies-Bouldin Index** (range: 0+, lower is better)
   - Ratio of within-cluster to between-cluster distances
   - Lower values indicate better separation

3. **Calinski-Harabasz Score** (range: 0+, higher is better)
   - Ratio of between-cluster to within-cluster dispersion
   - Higher values indicate more defined clusters

#### Qualitative Metrics:
- Cluster coherence (intra-cluster similarity)
- Semantic consistency of cluster keywords
- Topic separation quality

---

## 2. Results

### 2.1 Quantitative Comparison

| Configuration | Silhouette | Davies-Bouldin | Calinski-Harabasz | Clusters | Composite Score |
|--------------|------------|----------------|-------------------|----------|-----------------|
| **vs50_mc1_ep40** | **0.0916** | **4.3213** ✓ | **114.56** ✓ | 4 | **0.5157** ✓ |
| vs100_mc2_ep50 | 0.0838 | 4.8021 | 74.74 | 5 | 0.4033 |
| vs200_mc1_ep60 | **0.0987** ✓ | 4.9315 | 74.40 | 4 | 0.3963 |

**Key Observations:**
- vs50 configuration achieved the **best Davies-Bouldin Index** (4.32) indicating superior cluster separation
- vs50 configuration achieved the **best Calinski-Harabasz Score** (114.56) indicating better-defined clusters
- vs200 achieved the best Silhouette Score (0.099) but performed poorly on other metrics
- All configurations show relatively low Silhouette scores (< 0.1), indicating challenging data with overlapping semantic content

### 2.2 Cluster Analysis by Configuration

#### Configuration 1: vs50_mc1_ep40 (RECOMMENDED)
**Clusters:** 4
**Average Intra-Similarity:** 0.291

| Cluster | Size | Avg Distance | Similarity | Theme |
|---------|------|--------------|------------|-------|
| 0 | 397 | 13.62 | 0.280 | Pro models, iPhone Air discussion |
| 1 | 3,341 | 7.31 | **0.374** | General iPhone Pro/Pro Max discussion |
| 2 | 1,041 | 11.65 | 0.269 | Technical issues and questions |
| 3 | 217 | 17.41 | 0.243 | Meta posts (weekly threads, support) |

**Strengths:**
- Largest main cluster (3,341 posts) with highest coherence (0.374)
- Clear separation between technical discussions and product discussions
- Meta/administrative posts cleanly separated
- Most balanced cluster size distribution

#### Configuration 2: vs100_mc2_ep50
**Clusters:** 5
**Average Intra-Similarity:** 0.293

| Cluster | Size | Avg Distance | Similarity | Theme |
|---------|------|--------------|------------|-------|
| 0 | 689 | 15.01 | 0.235 | Technical issues |
| 2 | 352 | 17.21 | 0.238 | Pro models discussion |
| 3 | 3,112 | 7.72 | **0.398** | General discussion |
| 4 | 30 | 23.18 | 0.338 | Weekly threads (very small) |
| 5 | 813 | 13.25 | 0.256 | Feature/camera discussions |

**Strengths:**
- Highest intra-cluster similarity in main cluster (0.398)
- Good separation of technical vs. feature discussions

**Weaknesses:**
- Very small cluster 4 (only 30 posts) suggests over-segmentation
- Higher Davies-Bouldin Index (4.80) indicates some cluster overlap
- min_count=2 may have filtered useful rare words

#### Configuration 3: vs200_mc1_ep60
**Clusters:** 4
**Average Intra-Similarity:** 0.278

| Cluster | Size | Avg Distance | Similarity | Theme |
|---------|------|--------------|------------|-------|
| 0 | 67 | 24.85 | 0.253 | Weekly threads (very small) |
| 1 | 3,607 | 8.91 | 0.378 | General discussion |
| 3 | 882 | 15.50 | 0.236 | Pro model enthusiasm |
| 5 | 440 | 16.07 | 0.244 | Technical issues |

**Strengths:**
- Best Silhouette Score (0.099)
- Largest main cluster (3,607 posts)

**Weaknesses:**
- Very small cluster 0 (only 67 posts)
- Highest Davies-Bouldin Index (4.93) - worst cluster separation
- Lowest average intra-similarity (0.278)
- Higher dimensionality (200) may lead to sparse embeddings with limited data

---

## 3. Qualitative Analysis

### 3.1 Cluster Coherence Examination

Examining sample posts from each configuration's clusters reveals:

**vs50_mc1_ep40 - Most Coherent Clusters:**
- **Cluster 1** (main cluster): Highly coherent with consistent themes around iPhone Pro/Max features, colors, and general praise
- **Cluster 3** (meta posts): Cleanly separated weekly threads and support megathreads
- Clear topical separation visible in sample titles

**vs100_mc2_ep50 - Good Separation but Small Outlier:**
- Main cluster very coherent but creates tiny cluster (30 posts) suggesting instability
- May be overfitting to minor variations

**vs200_mc1_ep60 - Largest Spread:**
- Higher average distances to centroids indicate less compact clusters
- Large dimensionality creates sparse embeddings

### 3.2 Keyword Analysis

All configurations identify similar top keywords:
- "apple", "pro", "pro max", "iphone"

However, **vs50** shows better keyword distribution across clusters, with each cluster having more distinctive keyword patterns.

---

## 4. Composite Scoring Methodology

A weighted composite score was calculated for each configuration:

**Formula:**
```
Composite Score = 0.35 × Silhouette_norm + 0.25 × Davies-Bouldin_norm +
                  0.25 × Calinski-Harabasz_norm + 0.15 × Avg_Similarity
```

**Weights Rationale:**
- **Silhouette (35%):** Most important for overall cluster quality
- **Davies-Bouldin (25%):** Critical for cluster separation
- **Calinski-Harabasz (25%):** Important for cluster definition
- **Intra-Similarity (15%):** Confirms semantic coherence

**Results:**
1. **vs50_mc1_ep40:** 0.5157 ✓
2. vs100_mc2_ep50: 0.4033
3. vs200_mc1_ep60: 0.3963

---

## 5. Discussion

### 5.1 Why vs50_mc1_ep40 Performs Best

1. **Optimal Dimensionality:** 50 dimensions appears to be the "sweet spot" for this dataset
   - Large enough to capture semantic relationships
   - Small enough to avoid sparsity with 5,000 documents
   - Rule of thumb: vector size should be proportional to √(vocabulary size)

2. **Inclusive Vocabulary (min_count=1):**
   - Preserves rare technical terms and product names specific to iPhone discussions
   - Important for domain-specific clustering where rare words carry meaning

3. **Balanced Training (40 epochs):**
   - Sufficient training without overfitting
   - Achieves convergence for smaller embedding space

4. **Best Cluster Structure:**
   - Produces 4 meaningful, well-separated clusters
   - No micro-clusters (all clusters > 200 posts)
   - Best balance between separation and cohesion

### 5.2 Why Larger Embeddings Underperform

**vs100_mc2_ep50:**
- min_count=2 filters out rare but meaningful terms
- Creates unstable micro-cluster (30 posts)
- Higher dimensionality without corresponding benefit

**vs200_mc1_ep60:**
- Curse of dimensionality: embeddings become sparse
- 200 dimensions too large for 5,000 documents
- Requires significantly more training data to be effective
- Worst cluster separation (highest Davies-Bouldin Index)

### 5.3 Low Absolute Silhouette Scores

All configurations show low Silhouette scores (< 0.1), which is expected for this dataset because:

1. **Overlapping Topics:** Reddit posts about iPhones naturally discuss similar topics
2. **Natural Language Ambiguity:** Posts can belong to multiple themes
3. **User-Generated Content:** Inconsistent writing styles and mixed topics within posts
4. **Domain Specificity:** All posts are about iPhones, limiting inter-cluster variation

Despite low absolute values, **relative differences** between configurations are meaningful and consistent across multiple metrics.

---

## 6. Recommendations

### 6.1 Primary Recommendation

**Use Configuration 1: vector_size=50, min_count=1, epochs=40**

**Justification:**
- Highest composite score (0.5157)
- Best cluster separation (Davies-Bouldin: 4.32)
- Best cluster definition (Calinski-Harabasz: 114.56)
- Most balanced cluster sizes
- No unstable micro-clusters
- Good semantic coherence within clusters

### 6.2 When to Consider Alternatives

**Use vs100_mc2_ep50 if:**
- You have more training data (10,000+ documents)
- You want to filter out very rare words
- You need slightly higher intra-cluster similarity

**Use vs200_mc1_ep60 if:**
- You have significantly more data (50,000+ documents)
- You need very high-dimensional representations
- You're combining with other high-dimensional features

### 6.3 Future Improvements

1. **Collect More Data:** Larger datasets would benefit from higher-dimensional embeddings
2. **Hyperparameter Tuning:** Test intermediate values (vector_size=75)
3. **Ensemble Approaches:** Combine multiple Doc2Vec models
4. **Alternative Algorithms:** Compare with BERT, sentence transformers
5. **Domain-Specific Training:** Pre-train on larger iPhone/tech corpus

---

## 7. Conclusion

Through comprehensive quantitative and qualitative analysis of three Doc2Vec configurations, **Configuration 1 (vector_size=50, min_count=1, epochs=40)** emerged as the superior choice for clustering Reddit posts about iPhones.

**Key Takeaways:**

1. **Smaller is Better (for this dataset):** 50-dimensional embeddings outperformed larger alternatives, avoiding the curse of dimensionality while capturing sufficient semantic information.

2. **Keep Rare Words:** min_count=1 preserved important domain-specific terminology crucial for iPhone discussions.

3. **Balanced Metrics Matter:** The best configuration excelled across multiple evaluation metrics, not just one.

4. **Context is Critical:** The "best" configuration depends on dataset size, domain specificity, and clustering goals.

5. **Cluster Quality Over Quantity:** 4 well-defined clusters proved superior to 5 clusters with unstable micro-clusters.

This analysis demonstrates that careful configuration selection and comprehensive evaluation are essential for effective document clustering. While all three configurations produced meaningful clusters, the optimal choice balances embedding dimensionality with dataset size and domain requirements.

---

## Appendix A: Cluster Examples

### Configuration 1 (vs50_mc1_ep40) - Sample Posts

**Cluster 0 (Pro Model Discussions):**
- "Usually don't glaze iPhone…"
- "The Xiaomi 17 Pro is a great example of why specs ABSOLUTELY don't correlate to performance"
- "Long time Galaxy user going all in on Apple"

**Cluster 1 (General Enthusiasm):**
- "Apple should've let me done the colors"
- "iphone 17 pro color change purchased in Czech Republic"
- "Loving the 17 Pro Max"

**Cluster 2 (Technical Issues):**
- "Nine years of AppleCare+???"
- "Why is my phone ruining my photos like this?"
- "Every OS App Icon before 26!"

**Cluster 3 (Meta/Support):**
- "Weekly 'What Should I Buy' and Order/Shipping Thread"
- "Weekly iOS Battery Support Megathread"
- "Serious reason why Apple hasn't implemented that"

---

## Appendix B: Files Generated

1. **doc2vec_comparison_analysis.py** - Analysis script
2. **doc2vec_comparison_report.txt** - Summary report
3. **doc2vec_comparison_metrics.png** - Visualization comparing all metrics
4. **LAB8_FINAL_REPORT.md** - This comprehensive report

---

**End of Report**
