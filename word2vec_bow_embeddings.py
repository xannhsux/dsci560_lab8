#!/usr/bin/env python3
"""
Generate document embeddings by clustering Word2Vec word embeddings into bins
and computing normalized bag-of-bins vectors, then compare against Doc2Vec.

Usage examples:
    python word2vec_bow_embeddings.py --dimensions 50,100,300
    python word2vec_bow_embeddings.py --min-count 2 --epochs 30
"""

import argparse
import json
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def parse_dimensions(dim_string: str) -> List[int]:
    raw_values = [value.strip() for value in dim_string.split(",") if value.strip()]
    dimensions = []
    for value in raw_values:
        try:
            dim = int(value)
            if dim <= 0:
                raise ValueError
            dimensions.append(dim)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid dimension value: '{value}'")
    if not dimensions:
        raise argparse.ArgumentTypeError("At least one valid dimension is required")
    return dimensions


def load_posts(db_path: Path) -> List[Dict[str, str]]:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Could not find SQLite database at {db_path}. "
            "Run reddit_data_processor.py first to collect data."
        )

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, COALESCE(cleaned_content, '') AS cleaned_content,
                   COALESCE(title, '') AS title,
                   COALESCE(content, '') AS original_content
            FROM posts
            WHERE TRIM(COALESCE(cleaned_content, '') || COALESCE(title, '') || COALESCE(content, '')) != ''
            """
        )
        rows = cursor.fetchall()
    finally:
        connection.close()

    posts = []
    for row in rows:
        text_parts = [row["title"].strip(), row["cleaned_content"].strip()]
        fallback = row["original_content"].strip()
        text = " ".join(part for part in text_parts if part)
        if not text and fallback:
            text = fallback
        posts.append({"id": row["id"], "text": text})
    return posts


def tokenize_posts(posts: Sequence[Dict[str, str]]) -> List[List[str]]:
    return [simple_preprocess(post["text"], deacc=True) for post in posts]


def train_word2vec(
    tokenized_posts: Sequence[Sequence[str]],
    vector_size: int,
    min_count: int,
    epochs: int,
    window: int,
) -> Word2Vec:
    sentences = [tokens for tokens in tokenized_posts if tokens]
    if not sentences:
        raise RuntimeError("No tokens available to train Word2Vec. Check your dataset.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        epochs=epochs,
        negative=10,
    )
    logging.info(
        "Trained Word2Vec model with %d vocabulary words",
        len(model.wv.index_to_key),
    )
    return model


def cluster_word_vectors(
    model: Word2Vec, num_clusters: int, mini_batch: bool
) -> Tuple[Dict[str, int], int]:
    words = model.wv.index_to_key
    if not words:
        raise RuntimeError("Word2Vec vocabulary is empty; cannot cluster word vectors.")

    usable_clusters = min(num_clusters, len(words))
    if usable_clusters < num_clusters:
        logging.warning(
            "Requested %d clusters, but only %d unique words are available. "
            "Using %d clusters instead.",
            num_clusters,
            len(words),
            usable_clusters,
        )

    word_vectors = model.wv[words]
    if mini_batch:
        clusterer = MiniBatchKMeans(
            n_clusters=usable_clusters,
            random_state=42,
            n_init=10,
            batch_size=max(usable_clusters * 10, 100),
        )
    else:
        clusterer = KMeans(
            n_clusters=usable_clusters,
            random_state=42,
            n_init=10,
        )
    cluster_labels = clusterer.fit_predict(word_vectors)
    word_to_cluster = {word: int(label) for word, label in zip(words, cluster_labels)}
    return word_to_cluster, usable_clusters


def build_doc_vectors(
    posts: Sequence[Dict[str, str]],
    tokenized_posts: Sequence[Sequence[str]],
    word_to_cluster: Dict[str, int],
    num_clusters: int,
) -> Tuple[List[List[float]], List[str]]:
    vectors: List[List[float]] = []
    ids: List[str] = []
    for post, tokens in zip(posts, tokenized_posts):
        cluster_counts = np.zeros(num_clusters, dtype=float)
        hits = 0
        for token in tokens:
            cluster_id = word_to_cluster.get(token)
            if cluster_id is not None and cluster_id < num_clusters:
                cluster_counts[cluster_id] += 1
                hits += 1
        if hits > 0:
            cluster_counts /= hits
        vectors.append(cluster_counts.tolist())
        ids.append(post["id"])
    return vectors, ids


def train_doc2vec(
    tokenized_posts: Sequence[Sequence[str]], post_ids: Sequence[str], vector_size: int, epochs: int
) -> Tuple[List[List[float]], Doc2Vec]:
    documents = [
        TaggedDocument(words=tokens, tags=[pid])
        for pid, tokens in zip(post_ids, tokenized_posts)
        if tokens
    ]
    if not documents:
        raise RuntimeError("No tokenized documents available to train Doc2Vec.")
    model = Doc2Vec(
        vector_size=vector_size,
        epochs=epochs,
        min_count=1,
        workers=1,
        dm=1,
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = [model.dv[doc.tags[0]].tolist() for doc in documents]
    doc_ids = [doc.tags[0] for doc in documents]
    # Align order with input lists
    id_to_vector = dict(zip(doc_ids, vectors))
    ordered_vectors = [id_to_vector.get(pid, [0.0] * vector_size) for pid in post_ids]
    return ordered_vectors, model


def choose_cluster_count(num_docs: int) -> int:
    if num_docs < 4:
        return 2
    return max(2, min(10, int(round(math.sqrt(num_docs)))))


def evaluate_vectors(vectors: Sequence[Sequence[float]], n_clusters: int) -> Dict[str, float]:
    data = np.array(vectors, dtype=float)
    if len(data) < n_clusters or n_clusters < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(data)
    if len(set(labels)) < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    # Use cosine metric for silhouette to match Doc2Vec evaluation and assignment requirements
    silhouette = silhouette_score(data, labels, metric='cosine')
    calinski = calinski_harabasz_score(data, labels)
    davies = davies_bouldin_score(data, labels)
    return {
        "silhouette": float(silhouette),
        "calinski_harabasz": float(calinski),
        "davies_bouldin": float(davies),
    }


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_vectors(path: Path, vectors: Sequence[Sequence[float]], ids: Sequence[str]) -> None:
    ensure_directory(path.parent)
    np.save(path, np.array(vectors, dtype=float))
    metadata_path = path.with_suffix(".ids.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(ids, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Word2Vec bag-of-bins embeddings against Doc2Vec."
    )
    parser.add_argument(
        "--dimensions",
        type=parse_dimensions,
        default=[50, 100, 200],
        help="Comma-separated list of embedding dimensions to evaluate (default: 50,100,200)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("reddit_data") / "reddit_posts.db",
        help="Path to the SQLite database produced by reddit_data_processor.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reddit_data") / "embeddings",
        help="Directory to store generated embeddings and reports",
    )
    parser.add_argument(
        "--word2vec-size",
        type=int,
        default=200,
        help="Vector size for intermediate Word2Vec model (default: 200)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum word frequency for Word2Vec vocabulary (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for Word2Vec and Doc2Vec models (default: 30)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Context window size for Word2Vec (default: 5)",
    )
    parser.add_argument(
        "--mini-batch",
        action="store_true",
        help="Use MiniBatchKMeans instead of full KMeans for word clustering",
    )
    args = parser.parse_args()

    logging.info("Loading posts from %s", args.db_path)
    posts = load_posts(args.db_path)
    if len(posts) == 0:
        raise RuntimeError(
            "No posts were found in the database. Run reddit_data_processor.py to populate data."
        )

    tokenized_posts = tokenize_posts(posts)
    post_ids = [post["id"] for post in posts]
    logging.info("Prepared %d posts for embedding generation", len(posts))

    logging.info("Training shared Word2Vec model (size=%d)", args.word2vec_size)
    word2vec_model = train_word2vec(
        tokenized_posts,
        vector_size=args.word2vec_size,
        min_count=args.min_count,
        epochs=args.epochs,
        window=args.window,
    )

    ensure_directory(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for dimension in args.dimensions:
        logging.info("Processing dimension %d", dimension)

        word_clusters, actual_bins = cluster_word_vectors(
            word2vec_model, dimension, mini_batch=args.mini_batch
        )
        word_bins_path = args.output_dir / f"word2vec_bins_{dimension}.json"
        with open(word_bins_path, "w", encoding="utf-8") as handle:
            json.dump(word_clusters, handle, indent=2)

        bow_vectors, bow_ids = build_doc_vectors(posts, tokenized_posts, word_clusters, actual_bins)
        bow_path = args.output_dir / f"word2vec_bow_{dimension}.npy"
        save_vectors(bow_path, bow_vectors, bow_ids)

        cluster_count = choose_cluster_count(len(bow_vectors))
        bow_scores = evaluate_vectors(bow_vectors, cluster_count)

        doc2vec_vectors, doc2vec_model = train_doc2vec(
            tokenized_posts, post_ids, vector_size=dimension, epochs=args.epochs
        )
        doc2vec_path = args.output_dir / f"doc2vec_{dimension}.npy"
        save_vectors(doc2vec_path, doc2vec_vectors, post_ids)
        doc2vec_model_path = args.output_dir / f"doc2vec_{dimension}.model"
        doc2vec_model.save(str(doc2vec_model_path))
        doc2vec_scores = evaluate_vectors(doc2vec_vectors, cluster_count)

        summary[str(dimension)] = {
            "word2vec_bow": {
                "dimension": actual_bins,
                "clusters_evaluated": cluster_count,
                **bow_scores,
            },
            "doc2vec": {
                "dimension": dimension,
                "clusters_evaluated": cluster_count,
                **doc2vec_scores,
            },
        }

        logging.info(
            "Dimension %d complete | Word2Vec silhouette: %s | Doc2Vec silhouette: %s",
            dimension,
            f"{bow_scores['silhouette']:.4f}" if bow_scores["silhouette"] is not None else "N/A",
            f"{doc2vec_scores['silhouette']:.4f}" if doc2vec_scores["silhouette"] is not None else "N/A",
        )

    report_path = args.output_dir / f"embedding_comparison_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": timestamp,
                "dimensions": args.dimensions,
                "summary": summary,
            },
            handle,
            indent=2,
        )

    logging.info("Comparison report saved to %s", report_path)
    print("\n=== Comparison Summary ===")
    for dimension in args.dimensions:
        dim_key = str(dimension)
        if dim_key not in summary:
            continue
        word_scores = summary[dim_key]["word2vec_bow"]
        doc_scores = summary[dim_key]["doc2vec"]
        print(f"Dimension {dim_key}:")
        print(
            f"  Word2Vec bag-of-bins silhouette: "
            f"{word_scores['silhouette']:.4f}" if word_scores["silhouette"] is not None else "  Word2Vec bag-of-bins silhouette: N/A"
        )
        print(
            f"  Doc2Vec silhouette: "
            f"{doc_scores['silhouette']:.4f}" if doc_scores["silhouette"] is not None else "  Doc2Vec silhouette: N/A"
        )
        print(
            f"  Word2Vec Calinski-Harabasz: "
            f"{word_scores['calinski_harabasz']:.2f}" if word_scores["calinski_harabasz"] is not None else "  Word2Vec Calinski-Harabasz: N/A"
        )
        print(
            f"  Doc2Vec Calinski-Harabasz: "
            f"{doc_scores['calinski_harabasz']:.2f}" if doc_scores["calinski_harabasz"] is not None else "  Doc2Vec Calinski-Harabasz: N/A"
        )
        print(
            f"  Word2Vec Davies-Bouldin: "
            f"{word_scores['davies_bouldin']:.4f}" if word_scores["davies_bouldin"] is not None else "  Word2Vec Davies-Bouldin: N/A"
        )
        print(
            f"  Doc2Vec Davies-Bouldin: "
            f"{doc_scores['davies_bouldin']:.4f}" if doc_scores["davies_bouldin"] is not None else "  Doc2Vec Davies-Bouldin: N/A"
        )
        better = []
        if word_scores["silhouette"] and doc_scores["silhouette"]:
            if word_scores["silhouette"] > doc_scores["silhouette"]:
                better.append("Word2Vec silhouette")
            elif doc_scores["silhouette"] > word_scores["silhouette"]:
                better.append("Doc2Vec silhouette")
        if better:
            print("  Better metric(s): " + ", ".join(better))
        print()


if __name__ == "__main__":
    main()
