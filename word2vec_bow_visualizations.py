#!/usr/bin/env python3
"""
Generate PCA and t-SNE visualizations for Word2Vec bag-of-bins embeddings.

Examples:
    python word2vec_bow_visualizations.py --dimension 100
    python word2vec_bow_visualizations.py --dimension 50 --sample-size 1500
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def choose_cluster_count(num_docs: int) -> int:
    if num_docs < 4:
        return 2
    return max(2, min(12, int(round(math.sqrt(num_docs)))))


def load_embeddings(dimension: int, embeddings_dir: Path) -> np.ndarray:
    embedding_path = embeddings_dir / f"word2vec_bow_{dimension}.npy"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"无法找到 {embedding_path}，请先运行 word2vec_bow_embeddings.py 生成该维度的向量。"
        )
    vectors = np.load(embedding_path)
    if vectors.ndim != 2:
        raise ValueError(f"预期二维数组，得到形状 {vectors.shape}")
    logging.info("Loaded embeddings: %s (shape=%s)", embedding_path, vectors.shape)
    return vectors


def subsample_vectors(vectors: np.ndarray, sample_size: Optional[int]) -> np.ndarray:
    if sample_size is None or sample_size >= len(vectors):
        return vectors
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(vectors), size=sample_size, replace=False)
    logging.info("Subsampled %d / %d vectors for visualization", sample_size, len(vectors))
    return vectors[indices]


def cluster_vectors(vectors: np.ndarray, n_clusters: Optional[int]) -> np.ndarray:
    if n_clusters is None:
        n_clusters = choose_cluster_count(len(vectors))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    logging.info("Clustered vectors into %d groups for coloring", n_clusters)
    return labels


def plot_scatter(points: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    cmap = plt.cm.get_cmap("tab10")
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap=cmap, s=12, alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    ensure_directory(output_path.parent)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logging.info("Saved plot: %s", output_path)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_visualizations(
    vectors: np.ndarray,
    labels: np.ndarray,
    dimension: int,
    output_dir: Path,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(vectors)
    plot_scatter(
        pca_points,
        labels,
        f"Word2Vec Bag-of-Bins ({dimension}D) - PCA",
        output_dir / f"word2vec_bow_{dimension}_pca.png",
    )

    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    tsne_points = tsne.fit_transform(vectors)
    plot_scatter(
        tsne_points,
        labels,
        f"Word2Vec Bag-of-Bins ({dimension}D) - t-SNE",
        output_dir / f"word2vec_bow_{dimension}_tsne.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Word2Vec bag-of-bins embeddings.")
    parser.add_argument(
        "--dimension",
        type=int,
        required=True,
        help="维度（即词桶数量），例如 50 或 100。",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("reddit_data") / "embeddings",
        help="word2vec_bow_embeddings.py 输出的目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("word2vec_visualizations"),
        help="保存图像的目录。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="若帖子过多，可设定采样数量（默认 1500）。设置为 0 表示使用全部。",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="可自定义可视化时使用的聚类数量，未设置则自动估计。",
    )
    args = parser.parse_args()

    ensure_directory(args.output_dir)
    vectors = load_embeddings(args.dimension, args.embeddings_dir)

    if args.sample_size and args.sample_size > 0:
        vectors = subsample_vectors(vectors, args.sample_size)

    labels = cluster_vectors(vectors, args.clusters)
    build_visualizations(vectors, labels, args.dimension, args.output_dir)


if __name__ == "__main__":
    main()
