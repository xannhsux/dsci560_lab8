"""
PCA-based Dimensionality Reduction Analysis for Embeddings
Tests Word2Vec BoB and Doc2Vec with PCA reduction, focusing on WCSS minimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import json

sns.set_style("whitegrid")


def load_embeddings(method, dimension):
    """Load embeddings from file"""
    embeddings_dir = Path("reddit_data/embeddings")

    if method == "doc2vec":
        file_path = embeddings_dir / f"doc2vec_{dimension}.npy"
    else:  # word2vec_bob
        file_path = embeddings_dir / f"word2vec_bow_{dimension}.npy"

    return np.load(file_path)


def evaluate_clustering(embeddings, k, use_pca=False, pca_components=None):
    """
    Cluster embeddings and return comprehensive metrics

    Args:
        embeddings: Document embedding vectors
        k: Number of clusters
        use_pca: Whether to apply PCA before clustering
        pca_components: Number of PCA components (if use_pca=True)

    Returns:
        Dict with all metrics including WCSS
    """
    data = embeddings.copy()

    # Apply PCA if requested
    if use_pca and pca_components:
        pca = PCA(n_components=pca_components, random_state=42)
        data = pca.fit_transform(data)
        explained_var = pca.explained_variance_ratio_.sum()
    else:
        explained_var = 1.0

    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)

    # Calculate metrics
    wcss = kmeans.inertia_  # Within-Cluster Sum of Squares

    if len(set(labels)) >= 2:
        silhouette = silhouette_score(data, labels, metric='cosine')
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
    else:
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None

    return {
        'wcss': wcss,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz,
        'explained_variance': explained_var,
        'n_clusters': k,
        'dimensions_used': data.shape[1]
    }


def find_optimal_k_by_wcss(embeddings, min_k=2, max_k=15, use_pca=False, pca_components=None):
    """
    Find optimal k using elbow method on WCSS

    Returns:
        Dict with k_range and wcss values
    """
    wcss_values = []
    k_range = range(min_k, max_k + 1)

    for k in k_range:
        result = evaluate_clustering(embeddings, k, use_pca, pca_components)
        wcss_values.append(result['wcss'])

    # Find elbow using second derivative
    if len(wcss_values) >= 3:
        deltas = np.diff(wcss_values)
        second_deltas = np.diff(deltas)
        elbow_idx = np.argmax(second_deltas) + min_k + 1
    else:
        elbow_idx = min_k + len(wcss_values) // 2

    return {
        'k_range': list(k_range),
        'wcss_values': wcss_values,
        'optimal_k': elbow_idx
    }


def comprehensive_pca_analysis():
    """
    Comprehensive analysis testing:
    1. Original embeddings (no PCA)
    2. PCA-reduced embeddings at various dimensions
    """

    methods = ['word2vec_bob', 'doc2vec']
    original_dims = [50, 100, 200]
    pca_dims = [25, 50, 75, 100]  # PCA target dimensions
    k_clusters = 10  # Fixed k for comparison

    results = []

    print("="*80)
    print("PCA + EMBEDDING ANALYSIS (WCSS-focused)")
    print("="*80)

    for method in methods:
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")

        for orig_dim in original_dims:
            print(f"\n  Original Dimension: {orig_dim}D")
            print(f"  {'-'*70}")

            # Load embeddings
            embeddings = load_embeddings(method, orig_dim)

            # Test 1: No PCA (baseline)
            print(f"\n    [Baseline - No PCA]")
            baseline = evaluate_clustering(embeddings, k_clusters, use_pca=False)
            print(f"      WCSS: {baseline['wcss']:.2f}")
            print(f"      Silhouette: {baseline['silhouette']:.4f}" if baseline['silhouette'] else "      Silhouette: N/A")
            print(f"      Davies-Bouldin: {baseline['davies_bouldin']:.4f}" if baseline['davies_bouldin'] else "      Davies-Bouldin: N/A")
            print(f"      Calinski-Harabasz: {baseline['calinski_harabasz']:.2f}" if baseline['calinski_harabasz'] else "      Calinski-Harabasz: N/A")

            results.append({
                'method': method,
                'original_dim': orig_dim,
                'pca_dim': orig_dim,
                'pca_applied': False,
                **baseline
            })

            # Test 2: PCA to various dimensions (only if original_dim allows)
            for pca_dim in pca_dims:
                if pca_dim < orig_dim:  # Can only reduce, not increase
                    print(f"\n    [PCA to {pca_dim}D]")
                    pca_result = evaluate_clustering(embeddings, k_clusters,
                                                     use_pca=True, pca_components=pca_dim)
                    print(f"      WCSS: {pca_result['wcss']:.2f} (Variance: {pca_result['explained_variance']:.2%})")
                    print(f"      Silhouette: {pca_result['silhouette']:.4f}" if pca_result['silhouette'] else "      Silhouette: N/A")
                    print(f"      Davies-Bouldin: {pca_result['davies_bouldin']:.4f}" if pca_result['davies_bouldin'] else "      Davies-Bouldin: N/A")
                    print(f"      Calinski-Harabasz: {pca_result['calinski_harabasz']:.2f}" if pca_result['calinski_harabasz'] else "      Calinski-Harabasz: N/A")

                    results.append({
                        'method': method,
                        'original_dim': orig_dim,
                        'pca_dim': pca_dim,
                        'pca_applied': True,
                        **pca_result
                    })

    return pd.DataFrame(results)


def comprehensive_pca_analysis():
    """
    Comprehensive analysis testing:
    1. Original embeddings (no PCA)
    2. PCA-reduced embeddings at various dimensions
    """

    methods = ['word2vec_bob', 'doc2vec']
    original_dims = [50, 100, 200]
    pca_dims = [25, 50, 75, 100]  # PCA target dimensions
    k_clusters = 10  # Fixed k for comparison

    results = []

    print("="*80)
    print("PCA + EMBEDDING ANALYSIS (WCSS-focused)")
    print("="*80)

    for method in methods:
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")

        for orig_dim in original_dims:
            print(f"\n  Original Dimension: {orig_dim}D")
            print(f"  {'-'*70}")

            # Load embeddings
            embeddings = load_embeddings(method, orig_dim)

            # Test 1: No PCA (baseline)
            print(f"\n    [Baseline - No PCA]")
            baseline = evaluate_clustering(embeddings, k_clusters, use_pca=False)
            print(f"      WCSS: {baseline['wcss']:.2f}")
            print(f"      Silhouette: {baseline['silhouette']:.4f}" if baseline['silhouette'] else "      Silhouette: N/A")
            print(f"      Davies-Bouldin: {baseline['davies_bouldin']:.4f}" if baseline['davies_bouldin'] else "      Davies-Bouldin: N/A")
            print(f"      Calinski-Harabasz: {baseline['calinski_harabasz']:.2f}" if baseline['calinski_harabasz'] else "      Calinski-Harabasz: N/A")

            results.append({
                'method': method,
                'original_dim': orig_dim,
                'pca_dim': orig_dim,
                'pca_applied': False,
                **baseline
            })

            # Test 2: PCA to various dimensions (only if original_dim allows)
            for pca_dim in pca_dims:
                if pca_dim < orig_dim:  # Can only reduce, not increase
                    print(f"\n    [PCA to {pca_dim}D]")
                    pca_result = evaluate_clustering(embeddings, k_clusters,
                                                     use_pca=True, pca_components=pca_dim)
                    print(f"      WCSS: {pca_result['wcss']:.2f} (Variance: {pca_result['explained_variance']:.2%})")
                    print(f"      Silhouette: {pca_result['silhouette']:.4f}" if pca_result['silhouette'] else "      Silhouette: N/A")
                    print(f"      Davies-Bouldin: {pca_result['davies_bouldin']:.4f}" if pca_result['davies_bouldin'] else "      Davies-Bouldin: N/A")
                    print(f"      Calinski-Harabasz: {pca_result['calinski_harabasz']:.2f}" if pca_result['calinski_harabasz'] else "      Calinski-Harabasz: N/A")

                    results.append({
                        'method': method,
                        'original_dim': orig_dim,
                        'pca_dim': pca_dim,
                        'pca_applied': True,
                        **pca_result
                    })

    return pd.DataFrame(results)


def create_wcss_comparison_table(df):
    """Create a focused comparison table with WCSS as primary metric"""

    print("\n" + "="*80)
    print("WCSS COMPARISON TABLE (Lower is Better)")
    print("="*80)

    # Pivot table for WCSS
    pivot_wcss = df.pivot_table(
        values='wcss',
        index=['method', 'original_dim'],
        columns='pca_dim',
        aggfunc='first'
    )

    print("\nWCSS by Method, Original Dimension, and PCA Dimension:")
    print(pivot_wcss.to_string())

    # Find best configurations
    print("\n" + "="*80)
    print("BEST CONFIGURATIONS (by WCSS)")
    print("="*80)

    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        best_idx = method_df['wcss'].idxmin()
        best_row = method_df.loc[best_idx]

        print(f"\n{method.upper()}:")
        print(f"  Original Dim: {best_row['original_dim']}D")
        print(f"  PCA Dim: {best_row['pca_dim']}D")
        print(f"  PCA Applied: {best_row['pca_applied']}")
        print(f"  WCSS: {best_row['wcss']:.2f}")
        print(f"  Silhouette: {best_row['silhouette']:.4f}" if best_row['silhouette'] else "  Silhouette: N/A")
        print(f"  Davies-Bouldin: {best_row['davies_bouldin']:.4f}" if best_row['davies_bouldin'] else "  Davies-Bouldin: N/A")
        print(f"  Calinski-Harabasz: {best_row['calinski_harabasz']:.2f}" if best_row['calinski_harabasz'] else "  Calinski-Harabasz: N/A")


def create_wcss_visualization(df):
    """Create visualizations focused on WCSS"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA Impact on Clustering Quality (WCSS-Focused)', fontsize=16, fontweight='bold')

    # 1. WCSS by original dimension and PCA
    ax = axes[0, 0]
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        for orig_dim in method_df['original_dim'].unique():
            subset = method_df[method_df['original_dim'] == orig_dim]
            label = f"{method} ({orig_dim}D)"
            ax.plot(subset['pca_dim'], subset['wcss'], marker='o', label=label)

    ax.set_xlabel('PCA Dimensions', fontsize=11)
    ax.set_ylabel('WCSS (Lower is Better)', fontsize=11)
    ax.set_title('1. WCSS vs PCA Dimensions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Silhouette Scores
    ax = axes[0, 1]
    x_pos = 0
    labels = []
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        baseline = method_df[~method_df['pca_applied']]

        for _, row in baseline.iterrows():
            label = f"{method}\n{row['original_dim']}D"
            labels.append(label)
            color = 'skyblue' if method == 'doc2vec' else 'lightcoral'
            ax.bar(x_pos, row['silhouette'] if row['silhouette'] else 0, color=color, alpha=0.7)
            x_pos += 1

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('2. Silhouette Scores (No PCA)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # 3. Davies-Bouldin Index
    ax = axes[1, 0]
    baseline_df = df[~df['pca_applied']]
    x_pos = 0
    labels = []
    for _, row in baseline_df.iterrows():
        label = f"{row['method']}\n{row['original_dim']}D"
        labels.append(label)
        color = 'skyblue' if row['method'] == 'doc2vec' else 'lightcoral'
        ax.bar(x_pos, row['davies_bouldin'] if row['davies_bouldin'] else 0, color=color, alpha=0.7)
        x_pos += 1

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Davies-Bouldin Index (Lower is Better)', fontsize=11)
    ax.set_title('3. Davies-Bouldin Index (No PCA)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 4. Calinski-Harabasz Score
    ax = axes[1, 1]
    x_pos = 0
    labels = []
    for _, row in baseline_df.iterrows():
        label = f"{row['method']}\n{row['original_dim']}D"
        labels.append(label)
        color = 'skyblue' if row['method'] == 'doc2vec' else 'lightcoral'
        ax.bar(x_pos, row['calinski_harabasz'] if row['calinski_harabasz'] else 0, color=color, alpha=0.7)
        x_pos += 1

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calinski-Harabasz Score (Higher is Better)', fontsize=11)
    ax.set_title('4. Calinski-Harabasz Score (No PCA)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_wcss_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: pca_wcss_analysis.png")

    return fig


def main():
    """Run comprehensive PCA analysis"""

    # Run analysis
    df = comprehensive_pca_analysis()

    # Create comparison table
    create_wcss_comparison_table(df)

    # Create visualizations
    create_wcss_visualization(df)

    # Save detailed results
    output_file = 'pca_embedding_analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("  - WCSS values show clustering compactness")
    print("  - Lower WCSS = tighter, more compact clusters")
    print("  - PCA can reduce noise and improve clustering")
    print("  - Compare WCSS across configurations to find optimal setup")


if __name__ == "__main__":
    main()
