#!/usr/bin/env python3
"""
Comprehensive Embedding Comparison: Doc2Vec vs Word2Vec Bag-of-Bins

This script provides a complete comparison framework including:
- Quantitative metrics comparison (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- PCA visualizations for all configurations
- t-SNE visualizations for all configurations
- Side-by-side comparison plots
- Comprehensive summary report

Usage:
    python comprehensive_embedding_comparison.py
    python comprehensive_embedding_comparison.py --sample-size 1000  # For faster t-SNE
"""

import argparse
import json
import logging
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


class EmbeddingComparator:
    """Comprehensive comparison of Doc2Vec and Word2Vec Bag-of-Bins embeddings"""

    def __init__(
        self,
        embeddings_dir: Path = Path("reddit_data/embeddings"),
        output_dir: Path = Path("comparison_results"),
        dimensions: List[int] = [50, 100, 200]
    ):
        self.embeddings_dir = embeddings_dir
        self.output_dir = output_dir
        self.dimensions = dimensions
        self.results = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Initialized comparator with dimensions: {dimensions}")
        logging.info(f"Output directory: {output_dir}")

    def load_embeddings(self, method: str, dimension: int) -> Optional[np.ndarray]:
        """Load embeddings for a specific method and dimension"""
        if method == "doc2vec":
            file_path = self.embeddings_dir / f"doc2vec_{dimension}.npy"
        elif method == "word2vec_bow":
            file_path = self.embeddings_dir / f"word2vec_bow_{dimension}.npy"
        else:
            raise ValueError(f"Unknown method: {method}")

        if not file_path.exists():
            logging.warning(f"Embeddings file not found: {file_path}")
            return None

        embeddings = np.load(file_path)
        logging.info(f"Loaded {method} {dimension}D: {embeddings.shape}")
        return embeddings

    def choose_cluster_count(self, num_samples: int) -> int:
        """Automatically determine optimal cluster count"""
        if num_samples < 4:
            return 2
        return max(2, min(10, int(round(math.sqrt(num_samples)))))

    def cluster_and_evaluate(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> Dict:
        """Cluster embeddings and compute evaluation metrics"""
        if n_clusters is None:
            n_clusters = self.choose_cluster_count(len(embeddings))

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute metrics
        metrics = {
            'n_clusters': n_clusters,
            'n_samples': len(embeddings)
        }

        if len(set(labels)) >= 2:
            try:
                metrics['silhouette'] = silhouette_score(embeddings, labels, metric='cosine')
            except:
                metrics['silhouette'] = None

            try:
                metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
            except:
                metrics['davies_bouldin'] = None

            try:
                metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
            except:
                metrics['calinski_harabasz'] = None
        else:
            metrics['silhouette'] = None
            metrics['davies_bouldin'] = None
            metrics['calinski_harabasz'] = None

        metrics['labels'] = labels
        metrics['wcss'] = kmeans.inertia_

        return metrics

    def evaluate_all_configurations(self):
        """Evaluate all embedding configurations"""
        methods = ['doc2vec', 'word2vec_bow']

        for method in methods:
            for dimension in self.dimensions:
                config_name = f"{method}_{dimension}d"

                # Load embeddings
                embeddings = self.load_embeddings(method, dimension)
                if embeddings is None:
                    continue

                # Evaluate
                logging.info(f"Evaluating {config_name}...")
                metrics = self.cluster_and_evaluate(embeddings)

                # Store results
                self.results[config_name] = {
                    'method': method,
                    'dimension': dimension,
                    'embeddings': embeddings,
                    **metrics
                }

                logging.info(
                    f"  Silhouette: {metrics['silhouette']:.4f}"
                    if metrics['silhouette'] else "  Silhouette: N/A"
                )

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comprehensive comparison table"""
        data = []

        for config_name, result in self.results.items():
            row = {
                'Method': result['method'].replace('_', ' ').title(),
                'Dimension': result['dimension'],
                'Samples': result['n_samples'],
                'Clusters': result['n_clusters'],
                'Silhouette': result['silhouette'],
                'Davies-Bouldin': result['davies_bouldin'],
                'Calinski-Harabasz': result['calinski_harabasz'],
                'WCSS': result['wcss']
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by method and dimension
        df = df.sort_values(['Method', 'Dimension'])

        # Save to CSV
        csv_path = self.output_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logging.info(f"Saved comparison table: {csv_path}")

        return df

    def create_pca_grid(self, sample_size: Optional[int] = None):
        """Create grid of PCA visualizations for all configurations"""
        n_dims = len(self.dimensions)
        fig, axes = plt.subplots(2, n_dims, figsize=(6*n_dims, 12))
        fig.suptitle('PCA Visualization Comparison: Doc2Vec vs Word2Vec Bag-of-Bins',
                     fontsize=16, fontweight='bold', y=0.995)

        methods = ['doc2vec', 'word2vec_bow']
        method_names = ['Doc2Vec', 'Word2Vec Bag-of-Bins']

        for row, (method, method_name) in enumerate(zip(methods, method_names)):
            for col, dimension in enumerate(self.dimensions):
                ax = axes[row, col] if n_dims > 1 else axes[row]
                config_name = f"{method}_{dimension}d"

                if config_name not in self.results:
                    ax.text(0.5, 0.5, 'Data not available',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{method_name} {dimension}D')
                    continue

                result = self.results[config_name]
                embeddings = result['embeddings']
                labels = result['labels']

                # Subsample if requested
                if sample_size and len(embeddings) > sample_size:
                    indices = np.random.choice(len(embeddings), sample_size, replace=False)
                    embeddings = embeddings[indices]
                    labels = labels[indices]

                # Compute PCA
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(embeddings)

                # Plot
                scatter = ax.scatter(
                    pca_coords[:, 0],
                    pca_coords[:, 1],
                    c=labels,
                    cmap='tab10',
                    s=20,
                    alpha=0.6,
                    edgecolors='none'
                )

                # Add title with explained variance
                var_explained = pca.explained_variance_ratio_
                ax.set_title(
                    f'{method_name} {dimension}D\n' +
                    f'(Var: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})',
                    fontsize=11,
                    fontweight='bold'
                )
                ax.set_xlabel(f'PC1 ({var_explained[0]:.1%})', fontsize=9)
                ax.set_ylabel(f'PC2 ({var_explained[1]:.1%})', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Add metrics as text
                sil = result['silhouette']
                db = result['davies_bouldin']
                metrics_text = (
                    f"Sil: {sil:.3f}\n" if sil else "Sil: N/A\n"
                ) + (
                    f"DB: {db:.3f}" if db else "DB: N/A"
                )
                ax.text(
                    0.02, 0.98, metrics_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

        plt.tight_layout()
        output_path = self.output_dir / 'pca_comparison_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved PCA grid: {output_path}")

    def create_tsne_grid(self, sample_size: Optional[int] = 1500):
        """Create grid of t-SNE visualizations for all configurations"""
        n_dims = len(self.dimensions)
        fig, axes = plt.subplots(2, n_dims, figsize=(6*n_dims, 12))
        fig.suptitle('t-SNE Visualization Comparison: Doc2Vec vs Word2Vec Bag-of-Bins',
                     fontsize=16, fontweight='bold', y=0.995)

        methods = ['doc2vec', 'word2vec_bow']
        method_names = ['Doc2Vec', 'Word2Vec Bag-of-Bins']

        for row, (method, method_name) in enumerate(zip(methods, method_names)):
            for col, dimension in enumerate(self.dimensions):
                ax = axes[row, col] if n_dims > 1 else axes[row]
                config_name = f"{method}_{dimension}d"

                if config_name not in self.results:
                    ax.text(0.5, 0.5, 'Data not available',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{method_name} {dimension}D')
                    continue

                result = self.results[config_name]
                embeddings = result['embeddings']
                labels = result['labels']

                # Subsample for t-SNE (it's slow)
                if sample_size and len(embeddings) > sample_size:
                    indices = np.random.choice(len(embeddings), sample_size, replace=False)
                    embeddings = embeddings[indices]
                    labels = labels[indices]

                # Compute t-SNE
                logging.info(f"Computing t-SNE for {config_name} ({len(embeddings)} samples)...")
                perplexity = min(30, max(5, len(embeddings) // 10))
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perplexity,
                    max_iter=1000,
                    init='pca'
                )
                tsne_coords = tsne.fit_transform(embeddings)

                # Plot
                scatter = ax.scatter(
                    tsne_coords[:, 0],
                    tsne_coords[:, 1],
                    c=labels,
                    cmap='tab10',
                    s=20,
                    alpha=0.6,
                    edgecolors='none'
                )

                ax.set_title(f'{method_name} {dimension}D', fontsize=11, fontweight='bold')
                ax.set_xlabel('t-SNE Dim 1', fontsize=9)
                ax.set_ylabel('t-SNE Dim 2', fontsize=9)
                ax.grid(True, alpha=0.3)

                # Add metrics as text
                sil = result['silhouette']
                db = result['davies_bouldin']
                metrics_text = (
                    f"Sil: {sil:.3f}\n" if sil else "Sil: N/A\n"
                ) + (
                    f"DB: {db:.3f}" if db else "DB: N/A"
                )
                ax.text(
                    0.02, 0.98, metrics_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

        plt.tight_layout()
        output_path = self.output_dir / 'tsne_comparison_grid.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved t-SNE grid: {output_path}")

    def create_sidebyside_comparisons(self, sample_size: Optional[int] = 1500):
        """Create side-by-side comparisons for each dimension"""
        for dimension in self.dimensions:
            doc2vec_config = f"doc2vec_{dimension}d"
            word2vec_config = f"word2vec_bow_{dimension}d"

            if doc2vec_config not in self.results or word2vec_config not in self.results:
                logging.warning(f"Skipping {dimension}D comparison - missing data")
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle(
                f'Doc2Vec vs Word2Vec Bag-of-Bins ({dimension}D Comparison)',
                fontsize=16,
                fontweight='bold',
                y=0.995
            )

            # Get data
            doc2vec_result = self.results[doc2vec_config]
            word2vec_result = self.results[word2vec_config]

            # Subsample if needed
            doc2vec_emb = doc2vec_result['embeddings']
            doc2vec_labels = doc2vec_result['labels']
            word2vec_emb = word2vec_result['embeddings']
            word2vec_labels = word2vec_result['labels']

            if sample_size:
                if len(doc2vec_emb) > sample_size:
                    indices = np.random.choice(len(doc2vec_emb), sample_size, replace=False)
                    doc2vec_emb = doc2vec_emb[indices]
                    doc2vec_labels = doc2vec_labels[indices]

                if len(word2vec_emb) > sample_size:
                    indices = np.random.choice(len(word2vec_emb), sample_size, replace=False)
                    word2vec_emb = word2vec_emb[indices]
                    word2vec_labels = word2vec_labels[indices]

            # PCA - Doc2Vec
            ax = axes[0, 0]
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(doc2vec_emb)
            ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=doc2vec_labels,
                      cmap='tab10', s=20, alpha=0.6, edgecolors='none')
            var = pca.explained_variance_ratio_
            ax.set_title(f'Doc2Vec {dimension}D - PCA\n(PC1={var[0]:.1%}, PC2={var[1]:.1%})',
                        fontweight='bold')
            ax.set_xlabel(f'PC1 ({var[0]:.1%})')
            ax.set_ylabel(f'PC2 ({var[1]:.1%})')
            ax.grid(True, alpha=0.3)

            # PCA - Word2Vec
            ax = axes[0, 1]
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(word2vec_emb)
            ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=word2vec_labels,
                      cmap='tab10', s=20, alpha=0.6, edgecolors='none')
            var = pca.explained_variance_ratio_
            ax.set_title(f'Word2Vec BoW {dimension}D - PCA\n(PC1={var[0]:.1%}, PC2={var[1]:.1%})',
                        fontweight='bold')
            ax.set_xlabel(f'PC1 ({var[0]:.1%})')
            ax.set_ylabel(f'PC2 ({var[1]:.1%})')
            ax.grid(True, alpha=0.3)

            # t-SNE - Doc2Vec
            ax = axes[1, 0]
            logging.info(f"Computing t-SNE for Doc2Vec {dimension}D...")
            perplexity = min(30, max(5, len(doc2vec_emb) // 10))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca')
            tsne_coords = tsne.fit_transform(doc2vec_emb)
            ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=doc2vec_labels,
                      cmap='tab10', s=20, alpha=0.6, edgecolors='none')
            ax.set_title(f'Doc2Vec {dimension}D - t-SNE', fontweight='bold')
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.grid(True, alpha=0.3)

            # t-SNE - Word2Vec
            ax = axes[1, 1]
            logging.info(f"Computing t-SNE for Word2Vec BoW {dimension}D...")
            perplexity = min(30, max(5, len(word2vec_emb) // 10))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca')
            tsne_coords = tsne.fit_transform(word2vec_emb)
            ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=word2vec_labels,
                      cmap='tab10', s=20, alpha=0.6, edgecolors='none')
            ax.set_title(f'Word2Vec BoW {dimension}D - t-SNE', fontweight='bold')
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = self.output_dir / f'sidebyside_{dimension}d.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved side-by-side comparison: {output_path}")

    def create_metrics_comparison_charts(self):
        """Create bar charts comparing metrics across methods and dimensions"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metrics Comparison: Doc2Vec vs Word2Vec Bag-of-Bins',
                     fontsize=16, fontweight='bold')

        # Prepare data
        methods = []
        dimensions = []
        silhouettes = []
        davies_bouldins = []
        calinski_harabasz = []
        wcss = []

        for config_name, result in sorted(self.results.items()):
            methods.append(result['method'])
            dimensions.append(result['dimension'])
            silhouettes.append(result['silhouette'] if result['silhouette'] else 0)
            davies_bouldins.append(result['davies_bouldin'] if result['davies_bouldin'] else 0)
            calinski_harabasz.append(result['calinski_harabasz'] if result['calinski_harabasz'] else 0)
            wcss.append(result['wcss'])

        df = pd.DataFrame({
            'Method': methods,
            'Dimension': dimensions,
            'Silhouette': silhouettes,
            'Davies-Bouldin': davies_bouldins,
            'Calinski-Harabasz': calinski_harabasz,
            'WCSS': wcss
        })

        # Silhouette Score
        ax = axes[0, 0]
        x = np.arange(len(self.dimensions))
        width = 0.35
        doc2vec_sil = df[df['Method'] == 'doc2vec']['Silhouette'].values
        word2vec_sil = df[df['Method'] == 'word2vec_bow']['Silhouette'].values
        ax.bar(x - width/2, doc2vec_sil, width, label='Doc2Vec', color='skyblue')
        ax.bar(x + width/2, word2vec_sil, width, label='Word2Vec BoW', color='lightcoral')
        ax.set_title('Silhouette Score (Higher is Better)', fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dimensions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Davies-Bouldin Index
        ax = axes[0, 1]
        doc2vec_db = df[df['Method'] == 'doc2vec']['Davies-Bouldin'].values
        word2vec_db = df[df['Method'] == 'word2vec_bow']['Davies-Bouldin'].values
        ax.bar(x - width/2, doc2vec_db, width, label='Doc2Vec', color='skyblue')
        ax.bar(x + width/2, word2vec_db, width, label='Word2Vec BoW', color='lightcoral')
        ax.set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dimensions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Calinski-Harabasz Score
        ax = axes[1, 0]
        doc2vec_ch = df[df['Method'] == 'doc2vec']['Calinski-Harabasz'].values
        word2vec_ch = df[df['Method'] == 'word2vec_bow']['Calinski-Harabasz'].values
        ax.bar(x - width/2, doc2vec_ch, width, label='Doc2Vec', color='skyblue')
        ax.bar(x + width/2, word2vec_ch, width, label='Word2Vec BoW', color='lightcoral')
        ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dimensions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # WCSS
        ax = axes[1, 1]
        doc2vec_wcss = df[df['Method'] == 'doc2vec']['WCSS'].values
        word2vec_wcss = df[df['Method'] == 'word2vec_bow']['WCSS'].values
        ax.bar(x - width/2, doc2vec_wcss, width, label='Doc2Vec', color='skyblue')
        ax.bar(x + width/2, word2vec_wcss, width, label='Word2Vec BoW', color='lightcoral')
        ax.set_title('WCSS (Lower is Better)', fontweight='bold')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.dimensions)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved metrics comparison: {output_path}")

    def generate_summary_report(self, df: pd.DataFrame):
        """Generate comprehensive summary report in Markdown format"""
        report = f"""# Comprehensive Embedding Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Methods Compared:** Doc2Vec vs Word2Vec Bag-of-Bins
**Dimensions Tested:** {', '.join(map(str, self.dimensions))}

---

## Executive Summary

"""

        # Find best configurations
        best_sil = df.loc[df['Silhouette'].idxmax()]
        best_db = df.loc[df['Davies-Bouldin'].idxmin()]
        best_ch = df.loc[df['Calinski-Harabasz'].idxmax()]

        report += f"""
### Best Configurations by Metric

- **Best Silhouette Score:** {best_sil['Method']} {best_sil['Dimension']}D (Score: {best_sil['Silhouette']:.4f})
- **Best Davies-Bouldin Index:** {best_db['Method']} {best_db['Dimension']}D (Score: {best_db['Davies-Bouldin']:.4f})
- **Best Calinski-Harabasz Score:** {best_ch['Method']} {best_ch['Dimension']}D (Score: {best_ch['Calinski-Harabasz']:.2f})

---

## Quantitative Results

### Full Comparison Table

"""

        # Add formatted table
        report += df.to_markdown(index=False, floatfmt='.4f')

        report += """

---

## Metric Interpretations

### Silhouette Score
- **Range:** -1 to 1 (higher is better)
- **Interpretation:** Measures how similar objects are to their own cluster compared to other clusters
- **Guidelines:**
  - > 0.7: Strong structure
  - 0.5-0.7: Reasonable structure
  - 0.25-0.5: Weak structure
  - < 0.25: No substantial structure

### Davies-Bouldin Index
- **Range:** 0+ (lower is better)
- **Interpretation:** Ratio of within-cluster to between-cluster distances
- Lower values indicate better separation between clusters

### Calinski-Harabasz Score
- **Range:** 0+ (higher is better)
- **Interpretation:** Ratio of between-cluster to within-cluster dispersion
- Higher values indicate more defined, well-separated clusters

### WCSS (Within-Cluster Sum of Squares)
- **Range:** 0+ (lower is better)
- **Interpretation:** Total squared distance of each point to its cluster centroid
- Lower values indicate more compact clusters

---

## Visualization Files Generated

The following visualizations have been created in the `{self.output_dir}/` directory:

1. **pca_comparison_grid.png** - 6-panel PCA comparison (2 methods × 3 dimensions)
2. **tsne_comparison_grid.png** - 6-panel t-SNE comparison (2 methods × 3 dimensions)
3. **metrics_comparison.png** - Side-by-side bar charts of all metrics
4. **sidebyside_50d.png** - Direct comparison for 50D embeddings
5. **sidebyside_100d.png** - Direct comparison for 100D embeddings
6. **sidebyside_200d.png** - Direct comparison for 200D embeddings
7. **comparison_table.csv** - Raw data in CSV format

---

## Analysis by Dimension

"""

        for dim in self.dimensions:
            dim_df = df[df['Dimension'] == dim]
            doc2vec = dim_df[dim_df['Method'] == 'Doc2Vec'].iloc[0] if len(dim_df[dim_df['Method'] == 'Doc2Vec']) > 0 else None
            word2vec = dim_df[dim_df['Method'] == 'Word2Vec Bag-Of-Bins'].iloc[0] if len(dim_df[dim_df['Method'] == 'Word2Vec Bag-Of-Bins']) > 0 else None

            report += f"""
### {dim}D Comparison

"""

            if doc2vec is not None and word2vec is not None:
                report += f"""
| Metric | Doc2Vec | Word2Vec BoW | Winner |
|--------|---------|--------------|--------|
| Silhouette | {doc2vec['Silhouette']:.4f} | {word2vec['Silhouette']:.4f} | {'Doc2Vec' if doc2vec['Silhouette'] > word2vec['Silhouette'] else 'Word2Vec BoW'} |
| Davies-Bouldin | {doc2vec['Davies-Bouldin']:.4f} | {word2vec['Davies-Bouldin']:.4f} | {'Doc2Vec' if doc2vec['Davies-Bouldin'] < word2vec['Davies-Bouldin'] else 'Word2Vec BoW'} |
| Calinski-Harabasz | {doc2vec['Calinski-Harabasz']:.2f} | {word2vec['Calinski-Harabasz']:.2f} | {'Doc2Vec' if doc2vec['Calinski-Harabasz'] > word2vec['Calinski-Harabasz'] else 'Word2Vec BoW'} |
| WCSS | {doc2vec['WCSS']:.2f} | {word2vec['WCSS']:.2f} | {'Doc2Vec' if doc2vec['WCSS'] < word2vec['WCSS'] else 'Word2Vec BoW'} |

"""

        report += """
---

## Conclusion

Based on the comprehensive analysis across multiple metrics and dimensions, this report provides
quantitative evidence for comparing Doc2Vec and Word2Vec Bag-of-Bins approaches to document embedding.

The visualizations clearly show the cluster structure and separation achieved by each method, while
the numerical metrics provide objective measures of clustering quality.

**Recommendation:** Review both the quantitative metrics and visual cluster separation to determine
which method best suits your specific use case and requirements.

---

*Generated by comprehensive_embedding_comparison.py*
"""

        # Save report
        report_path = self.output_dir / 'comprehensive_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logging.info(f"Saved summary report: {report_path}")

        return report

    def run_complete_analysis(self, sample_size: Optional[int] = 1500):
        """Run the complete comparison analysis pipeline"""
        logging.info("="*80)
        logging.info("COMPREHENSIVE EMBEDDING COMPARISON")
        logging.info("="*80)

        # Step 1: Evaluate all configurations
        logging.info("\nStep 1: Evaluating all configurations...")
        self.evaluate_all_configurations()

        # Step 2: Generate comparison table
        logging.info("\nStep 2: Generating comparison table...")
        df = self.generate_comparison_table()
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))

        # Step 3: Create visualizations
        logging.info("\nStep 3: Creating visualizations...")

        logging.info("  Creating PCA grid...")
        self.create_pca_grid(sample_size=sample_size)

        logging.info("  Creating t-SNE grid...")
        self.create_tsne_grid(sample_size=sample_size)

        logging.info("  Creating side-by-side comparisons...")
        self.create_sidebyside_comparisons(sample_size=sample_size)

        logging.info("  Creating metrics comparison charts...")
        self.create_metrics_comparison_charts()

        # Step 4: Generate summary report
        logging.info("\nStep 4: Generating summary report...")
        self.generate_summary_report(df)

        # Done
        logging.info("\n" + "="*80)
        logging.info("ANALYSIS COMPLETE")
        logging.info("="*80)
        logging.info(f"\nAll results saved to: {self.output_dir}/")
        logging.info("\nGenerated files:")
        logging.info("  - comparison_table.csv")
        logging.info("  - pca_comparison_grid.png")
        logging.info("  - tsne_comparison_grid.png")
        logging.info("  - metrics_comparison.png")
        logging.info("  - sidebyside_50d.png")
        logging.info("  - sidebyside_100d.png")
        logging.info("  - sidebyside_200d.png")
        logging.info("  - comprehensive_report.md")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive comparison of Doc2Vec vs Word2Vec Bag-of-Bins embeddings'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=Path,
        default=Path('reddit_data/embeddings'),
        help='Directory containing embedding files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('comparison_results'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--dimensions',
        type=str,
        default='50,100,200',
        help='Comma-separated list of dimensions to compare (default: 50,100,200)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1500,
        help='Sample size for t-SNE visualization (default: 1500, use 0 for all data)'
    )

    args = parser.parse_args()

    # Parse dimensions
    dimensions = [int(d.strip()) for d in args.dimensions.split(',')]
    sample_size = args.sample_size if args.sample_size > 0 else None

    # Create comparator and run analysis
    comparator = EmbeddingComparator(
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        dimensions=dimensions
    )

    comparator.run_complete_analysis(sample_size=sample_size)


if __name__ == '__main__':
    main()
