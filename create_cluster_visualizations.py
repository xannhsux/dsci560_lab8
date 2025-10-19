#!/usr/bin/env python3
"""
Create comprehensive cluster visualizations including t-SNE and PCA plots.
Shows the clusters in 2D space along with detailed HTML report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from database_connection import SQLiteConnection
import matplotlib.patches as mpatches

VISUALIZATION_DIR = Path("visualizations")
VISUALIZATION_DIR.mkdir(exist_ok=True)

def load_cluster_data():
    """Load posts with embeddings and cluster assignments from SQLite"""
    db = SQLiteConnection()

    # Get posts with embeddings and cluster assignments
    results = db.execute_query("""
        SELECT
            p.id,
            p.title,
            p.cleaned_content,
            p.keywords,
            p.embedding,
            c.cluster_id,
            c.distance
        FROM posts p
        JOIN clusters c ON p.id = c.post_id
        WHERE p.embedding IS NOT NULL
        ORDER BY c.cluster_id, c.distance
    """, fetch='all')

    posts = []
    embeddings = []
    cluster_labels = []

    for row in results:
        post_id, title, content, keywords_json, embedding_json, cluster_id, distance = row

        try:
            embedding = np.array(json.loads(embedding_json))
            keywords = json.loads(keywords_json) if keywords_json else []
        except:
            continue

        posts.append({
            'id': post_id,
            'title': title or '',
            'content': content or '',
            'keywords': keywords,
            'cluster_id': cluster_id,
            'distance': distance
        })
        embeddings.append(embedding)
        cluster_labels.append(cluster_id)

    db.close()

    return posts, np.array(embeddings), np.array(cluster_labels)


def create_tsne_visualization(embeddings, labels, posts):
    """Create t-SNE visualization of clusters"""
    print("[TSNE] Computing t-SNE projection (this may take a moment)...")

    # Use perplexity based on dataset size
    n_samples = len(embeddings)
    perplexity = min(30, max(5, n_samples // 10))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique clusters and create color map
    unique_clusters = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Plot each cluster
    for idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_points = embeddings_2d[mask]

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[idx]],
            label=f'Cluster {cluster_id} ({np.sum(mask)} posts)',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_title('t-SNE Visualization of Reddit Post Clusters\n(Doc2Vec embeddings)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = VISUALIZATION_DIR / 'tsne_clusters.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved t-SNE visualization: {output_file}")
    return output_file


def create_pca_visualization(embeddings, labels, posts):
    """Create PCA visualization of clusters"""
    print("[PCA] Computing PCA projection...")

    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique clusters and create color map
    unique_clusters = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Plot each cluster
    for idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_points = embeddings_2d[mask]

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[idx]],
            label=f'Cluster {cluster_id} ({np.sum(mask)} posts)',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    # Add explained variance to title
    explained_var = pca.explained_variance_ratio_
    ax.set_title(f'PCA Visualization of Reddit Post Clusters\n' +
                 f'(Explained variance: PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%})',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = VISUALIZATION_DIR / 'pca_clusters.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved PCA visualization: {output_file}")
    return output_file


def create_combined_visualization(embeddings, labels, posts):
    """Create side-by-side t-SNE and PCA visualization"""
    print("[COMBINED] Creating combined t-SNE + PCA visualization...")

    # Compute both projections
    n_samples = len(embeddings)
    perplexity = min(30, max(5, n_samples // 10))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    tsne_2d = tsne.fit_transform(embeddings)

    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(embeddings)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Get unique clusters and create color map
    unique_clusters = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Plot t-SNE
    for idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_points = tsne_2d[mask]

        ax1.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[idx]],
            label=f'Cluster {cluster_id} ({np.sum(mask)} posts)',
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax1.set_title('t-SNE Projection', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11, fontweight='bold')
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot PCA
    for idx, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_points = pca_2d[mask]

        ax2.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[idx]],
            label=f'Cluster {cluster_id} ({np.sum(mask)} posts)',
            s=80,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    explained_var = pca.explained_variance_ratio_
    ax2.set_title(f'PCA Projection (PC1={explained_var[0]:.1%}, PC2={explained_var[1]:.1%})',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11, fontweight='bold')
    ax2.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Cluster Visualization: t-SNE vs PCA Dimensionality Reduction',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = VISUALIZATION_DIR / 'combined_tsne_pca.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved combined visualization: {output_file}")
    return output_file


def main():
    print("=" * 70)
    print("CLUSTER VISUALIZATION GENERATOR")
    print("=" * 70)

    print("\n[DATA] Loading cluster data from SQLite database...")
    posts, embeddings, labels = load_cluster_data()

    if len(posts) == 0:
        print("[ERROR] No clustered posts found in database!")
        return

    print(f"[SUCCESS] Loaded {len(posts)} posts across {len(np.unique(labels))} clusters")
    print(f"[INFO] Embedding dimension: {embeddings.shape[1]}")

    # Display cluster distribution
    print("\n[INFO] Cluster distribution:")
    for cluster_id in sorted(np.unique(labels)):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} posts")

    # Display features used for clustering
    print("\n[INFO] Features used for clustering:")
    print("  • Doc2Vec embeddings (100 dimensions)")
    print("  • Semantic representation of post title + content")
    print("  • Captures word context and meaning")
    print("\n[INFO] Additional features extracted (not used for clustering):")
    print("  • TF-IDF embeddings (300 features)")
    print("  • Numerical features: score, num_comments, text_length, etc.")
    print("  • Keywords, topics, URLs, mentions, hashtags")

    # Create visualizations
    print("\n[GENERATING] Creating visualizations...")
    print("-" * 70)

    tsne_file = create_tsne_visualization(embeddings, labels, posts)
    pca_file = create_pca_visualization(embeddings, labels, posts)
    combined_file = create_combined_visualization(embeddings, labels, posts)

    print("\n" + "=" * 70)
    print("[SUCCESS] All visualizations created!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  • t-SNE: {tsne_file}")
    print(f"  • PCA: {pca_file}")
    print(f"  • Combined: {combined_file}")
    print(f"\nTo view: open {VISUALIZATION_DIR}/")


if __name__ == '__main__':
    main()