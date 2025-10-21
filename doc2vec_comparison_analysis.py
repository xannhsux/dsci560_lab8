"""
Comprehensive Doc2Vec Configuration Comparison Analysis
Compares three different Doc2Vec configurations for Reddit post clustering
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Doc2VecComparison:
    def __init__(self, data_dir="reddit_data", session_id="20251019_005015"):
        """Initialize comparison with data directory and session ID"""
        self.data_dir = Path(data_dir)
        self.session_id = session_id

        # Three configurations to compare
        self.configs = [
            {'name': 'vs50_mc1_ep40', 'vector_size': 50, 'min_count': 1, 'epochs': 40},
            {'name': 'vs100_mc2_ep50', 'vector_size': 100, 'min_count': 2, 'epochs': 50},
            {'name': 'vs200_mc1_ep60', 'vector_size': 200, 'min_count': 1, 'epochs': 60}
        ]

        self.results = {}

    def load_configuration_data(self, config_name):
        """Load all necessary data for a configuration"""
        base_path = f"iphone_{self.session_id}_{config_name}"

        # Load cluster assignments
        clusters_file = self.data_dir / "metadata" / f"{base_path}_clusters.json"
        with open(clusters_file, 'r') as f:
            cluster_data = json.load(f)

        # Load processed data with embeddings
        data_file = self.data_dir / "processed" / f"{base_path}.pkl"
        with open(data_file, 'rb') as f:
            posts_data = pickle.load(f)

        # Load metadata
        meta_file = self.data_dir / "metadata" / f"{base_path}_meta.json"
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        return {
            'cluster_data': cluster_data,
            'posts_data': posts_data,
            'metadata': metadata,
            'config_name': config_name
        }

    def extract_embeddings_and_labels(self, posts_data, cluster_assignments):
        """Extract embeddings and cluster labels from data"""
        # Create mapping from post_id to cluster_id
        post_to_cluster = {item['post_id']: item['cluster_id']
                          for item in cluster_assignments}

        embeddings = []
        labels = []
        post_ids = []

        for post in posts_data:
            if post['id'] in post_to_cluster and post.get('embedding'):
                embeddings.append(post['embedding'])
                labels.append(post_to_cluster[post['id']])
                post_ids.append(post['id'])

        return np.array(embeddings), np.array(labels), post_ids

    def calculate_cluster_metrics(self, embeddings, labels):
        """Calculate comprehensive cluster quality metrics"""
        if len(embeddings) < 2 or len(np.unique(labels)) < 2:
            return None

        metrics = {}

        # Silhouette Score: measures how similar objects are to their own cluster
        # compared to other clusters. Range: [-1, 1], higher is better
        # > 0.7: Strong structure
        # 0.5-0.7: Reasonable structure
        # 0.25-0.5: Weak structure
        # < 0.25: No substantial structure
        try:
            metrics['silhouette_score'] = silhouette_score(embeddings, labels, metric='cosine')
        except:
            metrics['silhouette_score'] = None

        # Davies-Bouldin Index: ratio of within-cluster to between-cluster distances
        # Lower is better (0 is perfect)
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        except:
            metrics['davies_bouldin'] = None

        # Calinski-Harabasz Score: ratio of between-cluster to within-cluster dispersion
        # Higher is better
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
        except:
            metrics['calinski_harabasz'] = None

        return metrics

    def calculate_cluster_statistics(self, embeddings, labels, posts_data, post_ids):
        """Calculate detailed statistics for each cluster"""
        unique_labels = np.unique(labels)
        cluster_stats = []

        # Create post lookup
        post_lookup = {p['id']: p for p in posts_data}

        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_post_ids = [post_ids[i] for i in range(len(post_ids)) if cluster_mask[i]]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate distances from centroid
            distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]

            # Calculate intra-cluster cosine similarity
            if len(cluster_embeddings) > 1:
                sim_matrix = cosine_similarity(cluster_embeddings)
                # Get upper triangle (excluding diagonal)
                triu_indices = np.triu_indices_from(sim_matrix, k=1)
                avg_similarity = np.mean(sim_matrix[triu_indices]) if len(triu_indices[0]) > 0 else 0
            else:
                avg_similarity = 1.0

            # Extract keywords from posts in this cluster
            all_keywords = []
            all_topics = []
            cluster_posts = []

            for pid in cluster_post_ids:
                if pid in post_lookup:
                    post = post_lookup[pid]
                    all_keywords.extend(post.get('keywords', []))
                    all_topics.extend(post.get('topics', []))
                    cluster_posts.append({
                        'id': pid,
                        'title': post.get('title', ''),
                        'cleaned_content': post.get('cleaned_content', '')[:200]
                    })

            # Get top keywords and topics
            keyword_counter = Counter(all_keywords)
            topic_counter = Counter(all_topics)

            stats = {
                'cluster_id': int(cluster_id),
                'size': int(np.sum(cluster_mask)),
                'avg_distance_to_centroid': float(np.mean(distances)),
                'std_distance_to_centroid': float(np.std(distances)),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances)),
                'avg_intra_similarity': float(avg_similarity),
                'top_keywords': keyword_counter.most_common(10),
                'top_topics': topic_counter.most_common(10),
                'sample_posts': cluster_posts[:5]  # Sample posts for qualitative analysis
            }

            cluster_stats.append(stats)

        return cluster_stats

    def calculate_inter_cluster_distances(self, embeddings, labels):
        """Calculate distances between cluster centroids"""
        unique_labels = np.unique(labels)
        centroids = []

        for label in unique_labels:
            cluster_embeddings = embeddings[labels == label]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)

        # Calculate pairwise distances
        n_clusters = len(centroids)
        distances = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j:
                    distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        return distances, centroids

    def analyze_all_configurations(self):
        """Run complete analysis on all configurations"""
        print("="*80)
        print("DOC2VEC CONFIGURATION COMPARISON ANALYSIS")
        print("="*80)

        for config in self.configs:
            config_name = config['name']
            print(f"\n{'='*80}")
            print(f"Analyzing: {config_name}")
            print(f"Vector Size: {config['vector_size']}, Min Count: {config['min_count']}, Epochs: {config['epochs']}")
            print(f"{'='*80}")

            # Load data
            data = self.load_configuration_data(config_name)

            # Extract embeddings and labels
            embeddings, labels, post_ids = self.extract_embeddings_and_labels(
                data['posts_data'],
                data['cluster_data']['assignments']
            )

            print(f"Loaded {len(embeddings)} posts with embeddings")
            print(f"Number of clusters: {len(np.unique(labels))}")

            # Calculate metrics
            metrics = self.calculate_cluster_metrics(embeddings, labels)

            # Calculate cluster statistics
            cluster_stats = self.calculate_cluster_statistics(
                embeddings, labels, data['posts_data'], post_ids
            )

            # Calculate inter-cluster distances
            inter_distances, centroids = self.calculate_inter_cluster_distances(
                embeddings, labels
            )

            # Store results
            self.results[config_name] = {
                'config': config,
                'metrics': metrics,
                'cluster_stats': cluster_stats,
                'inter_cluster_distances': inter_distances,
                'centroids': centroids,
                'n_clusters': len(np.unique(labels)),
                'n_posts': len(embeddings),
                'embeddings': embeddings,
                'labels': labels
            }

            # Print summary
            print(f"\nCluster Quality Metrics:")
            if metrics:
                print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}" if metrics['silhouette_score'] else "  Silhouette Score: N/A")
                print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] else "  Davies-Bouldin Index: N/A")
                print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}" if metrics['calinski_harabasz'] else "  Calinski-Harabasz Score: N/A")

            print(f"\nCluster Size Distribution:")
            for stat in cluster_stats:
                print(f"  Cluster {stat['cluster_id']}: {stat['size']} posts "
                      f"(avg dist: {stat['avg_distance_to_centroid']:.2f}, "
                      f"avg sim: {stat['avg_intra_similarity']:.3f})")

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("QUANTITATIVE COMPARISON")
        print("="*80)

        # Create comparison table
        comparison_data = []
        for config_name, result in self.results.items():
            config = result['config']
            metrics = result['metrics']

            row = {
                'Configuration': config_name,
                'Vector Size': config['vector_size'],
                'Min Count': config['min_count'],
                'Epochs': config['epochs'],
                'Num Clusters': result['n_clusters'],
                'Silhouette': metrics['silhouette_score'] if metrics and metrics['silhouette_score'] else 0,
                'Davies-Bouldin': metrics['davies_bouldin'] if metrics and metrics['davies_bouldin'] else 0,
                'Calinski-Harabasz': metrics['calinski_harabasz'] if metrics and metrics['calinski_harabasz'] else 0
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        print("\n" + df.to_string(index=False))

        # Determine best configuration
        print("\n" + "="*80)
        print("RANKING BY METRICS")
        print("="*80)

        # Silhouette (higher is better)
        best_silhouette = df.loc[df['Silhouette'].idxmax()]
        print(f"\nBest Silhouette Score: {best_silhouette['Configuration']} ({best_silhouette['Silhouette']:.4f})")

        # Davies-Bouldin (lower is better)
        best_davies = df.loc[df['Davies-Bouldin'].idxmin()]
        print(f"Best Davies-Bouldin Index: {best_davies['Configuration']} ({best_davies['Davies-Bouldin']:.4f})")

        # Calinski-Harabasz (higher is better)
        best_calinski = df.loc[df['Calinski-Harabasz'].idxmax()]
        print(f"Best Calinski-Harabasz Score: {best_calinski['Configuration']} ({best_calinski['Calinski-Harabasz']:.2f})")

        return df

    def generate_cluster_coherence_analysis(self):
        """Analyze and compare cluster coherence across configurations"""
        print("\n" + "="*80)
        print("CLUSTER COHERENCE ANALYSIS")
        print("="*80)

        for config_name, result in self.results.items():
            print(f"\n{'-'*80}")
            print(f"Configuration: {config_name}")
            print(f"{'-'*80}")

            cluster_stats = result['cluster_stats']

            for stat in cluster_stats:
                print(f"\nCluster {stat['cluster_id']} ({stat['size']} posts):")
                print(f"  Cohesion (avg intra-similarity): {stat['avg_intra_similarity']:.3f}")
                print(f"  Compactness (avg distance): {stat['avg_distance_to_centroid']:.2f} ± {stat['std_distance_to_centroid']:.2f}")
                print(f"  Top Keywords: {', '.join([kw for kw, _ in stat['top_keywords'][:5]])}")
                print(f"  Top Topics: {', '.join([t for t, _ in stat['top_topics'][:5]])}")

                print(f"  Sample Post Titles:")
                for i, post in enumerate(stat['sample_posts'][:3], 1):
                    title = post['title'] if post['title'] else post['cleaned_content'][:80]
                    print(f"    {i}. {title}")

    def visualize_comparison(self):
        """Create visualizations comparing the configurations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Prepare data for visualization
        config_names = list(self.results.keys())

        # 1. Silhouette Scores
        silhouette_scores = [self.results[c]['metrics']['silhouette_score']
                            if self.results[c]['metrics'] and self.results[c]['metrics']['silhouette_score']
                            else 0 for c in config_names]
        axes[0, 0].bar(config_names, silhouette_scores, color='skyblue')
        axes[0, 0].set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Add reference lines for silhouette interpretation
        axes[0, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Strong')
        axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Reasonable')
        axes[0, 0].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Weak')
        axes[0, 0].legend(fontsize=8)

        # 2. Davies-Bouldin Index
        davies_scores = [self.results[c]['metrics']['davies_bouldin']
                        if self.results[c]['metrics'] and self.results[c]['metrics']['davies_bouldin']
                        else 0 for c in config_names]
        axes[0, 1].bar(config_names, davies_scores, color='lightcoral')
        axes[0, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Calinski-Harabasz Score
        calinski_scores = [self.results[c]['metrics']['calinski_harabasz']
                          if self.results[c]['metrics'] and self.results[c]['metrics']['calinski_harabasz']
                          else 0 for c in config_names]
        axes[0, 2].bar(config_names, calinski_scores, color='lightgreen')
        axes[0, 2].set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(axis='y', alpha=0.3)

        # 4. Cluster Size Distribution
        for config_name in config_names:
            cluster_stats = self.results[config_name]['cluster_stats']
            sizes = [stat['size'] for stat in cluster_stats]
            cluster_ids = [stat['cluster_id'] for stat in cluster_stats]
            axes[1, 0].plot(cluster_ids, sizes, marker='o', label=config_name)

        axes[1, 0].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Posts')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 5. Average Intra-Cluster Similarity
        for config_name in config_names:
            cluster_stats = self.results[config_name]['cluster_stats']
            similarities = [stat['avg_intra_similarity'] for stat in cluster_stats]
            cluster_ids = [stat['cluster_id'] for stat in cluster_stats]
            axes[1, 1].plot(cluster_ids, similarities, marker='s', label=config_name)

        axes[1, 1].set_title('Average Intra-Cluster Similarity', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Cosine Similarity')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        # 6. Average Distance to Centroid
        for config_name in config_names:
            cluster_stats = self.results[config_name]['cluster_stats']
            distances = [stat['avg_distance_to_centroid'] for stat in cluster_stats]
            cluster_ids = [stat['cluster_id'] for stat in cluster_stats]
            axes[1, 2].plot(cluster_ids, distances, marker='^', label=config_name)

        axes[1, 2].set_title('Average Distance to Centroid', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Cluster ID')
        axes[1, 2].set_ylabel('Distance')
        axes[1, 2].legend()
        axes[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('doc2vec_comparison_metrics.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: doc2vec_comparison_metrics.png")

        return fig

    def generate_final_recommendation(self):
        """Generate final recommendation based on all analyses"""
        print("\n" + "="*80)
        print("FINAL RECOMMENDATION")
        print("="*80)

        # Collect all metrics
        config_scores = {}

        for config_name, result in self.results.items():
            metrics = result['metrics']
            if not metrics:
                continue

            # Normalize metrics to 0-1 scale and compute composite score
            # Silhouette: already in [-1, 1], normalize to [0, 1]
            silhouette_norm = (metrics['silhouette_score'] + 1) / 2 if metrics['silhouette_score'] else 0

            # Davies-Bouldin: lower is better, invert and normalize
            # Typical range is 0-3, but we'll use the observed range
            davies_vals = [self.results[c]['metrics']['davies_bouldin']
                          for c in self.results.keys()
                          if self.results[c]['metrics'] and self.results[c]['metrics']['davies_bouldin']]
            if davies_vals:
                max_davies = max(davies_vals)
                davies_norm = 1 - (metrics['davies_bouldin'] / max_davies) if metrics['davies_bouldin'] else 0
            else:
                davies_norm = 0

            # Calinski-Harabasz: higher is better, normalize
            calinski_vals = [self.results[c]['metrics']['calinski_harabasz']
                            for c in self.results.keys()
                            if self.results[c]['metrics'] and self.results[c]['metrics']['calinski_harabasz']]
            if calinski_vals:
                max_calinski = max(calinski_vals)
                calinski_norm = metrics['calinski_harabasz'] / max_calinski if metrics['calinski_harabasz'] else 0
            else:
                calinski_norm = 0

            # Calculate average intra-cluster similarity
            cluster_stats = result['cluster_stats']
            avg_similarity = np.mean([stat['avg_intra_similarity'] for stat in cluster_stats])

            # Composite score (weighted average)
            composite_score = (
                0.35 * silhouette_norm +      # Silhouette is most important
                0.25 * davies_norm +           # Davies-Bouldin
                0.25 * calinski_norm +         # Calinski-Harabasz
                0.15 * avg_similarity          # Intra-cluster similarity
            )

            config_scores[config_name] = {
                'composite_score': composite_score,
                'silhouette_norm': silhouette_norm,
                'davies_norm': davies_norm,
                'calinski_norm': calinski_norm,
                'avg_similarity': avg_similarity,
                'n_clusters': result['n_clusters']
            }

        # Sort by composite score
        ranked_configs = sorted(config_scores.items(),
                               key=lambda x: x[1]['composite_score'],
                               reverse=True)

        print("\nRanked Configurations (by composite score):")
        print("-" * 80)
        for rank, (config_name, scores) in enumerate(ranked_configs, 1):
            config = self.results[config_name]['config']
            print(f"\n{rank}. {config_name}")
            print(f"   Vector Size: {config['vector_size']}, Min Count: {config['min_count']}, Epochs: {config['epochs']}")
            print(f"   Composite Score: {scores['composite_score']:.4f}")
            print(f"   Number of Clusters: {scores['n_clusters']}")
            print(f"   Component Scores:")
            print(f"     - Silhouette (normalized): {scores['silhouette_norm']:.4f}")
            print(f"     - Davies-Bouldin (normalized): {scores['davies_norm']:.4f}")
            print(f"     - Calinski-Harabasz (normalized): {scores['calinski_norm']:.4f}")
            print(f"     - Avg Intra-Similarity: {scores['avg_similarity']:.4f}")

        # Best configuration
        best_config_name, best_scores = ranked_configs[0]
        best_config = self.results[best_config_name]['config']

        print("\n" + "="*80)
        print("RECOMMENDED CONFIGURATION")
        print("="*80)
        print(f"\nBest Configuration: {best_config_name}")
        print(f"  - Vector Size: {best_config['vector_size']}")
        print(f"  - Min Count: {best_config['min_count']}")
        print(f"  - Epochs: {best_config['epochs']}")
        print(f"  - Composite Score: {best_scores['composite_score']:.4f}")
        print(f"  - Number of Clusters: {best_scores['n_clusters']}")

        print(f"\nRationale:")
        print(f"  This configuration achieved the highest composite score based on:")
        print(f"  1. Cluster separation and cohesion (Silhouette Score)")
        print(f"  2. Cluster compactness (Davies-Bouldin Index)")
        print(f"  3. Cluster definition (Calinski-Harabasz Score)")
        print(f"  4. Semantic similarity within clusters (Intra-cluster similarity)")

        # Qualitative observations
        print(f"\nQualitative Observations:")
        best_cluster_stats = self.results[best_config_name]['cluster_stats']

        print(f"  - Produced {best_scores['n_clusters']} well-defined clusters")
        print(f"  - Average intra-cluster similarity: {best_scores['avg_similarity']:.3f}")

        # Show cluster themes
        print(f"\n  Cluster Themes:")
        for stat in best_cluster_stats:
            top_keywords = ', '.join([kw for kw, _ in stat['top_keywords'][:3]])
            print(f"    Cluster {stat['cluster_id']} ({stat['size']} posts): {top_keywords}")

        return best_config_name, best_scores


def main():
    """Run complete Doc2Vec configuration comparison"""
    # Initialize comparison
    comparator = Doc2VecComparison()

    # Run analysis
    comparator.analyze_all_configurations()

    # Generate comparison report
    df = comparator.generate_comparison_report()

    # Analyze cluster coherence
    comparator.generate_cluster_coherence_analysis()

    # Create visualizations
    comparator.visualize_comparison()

    # Generate final recommendation
    best_config, scores = comparator.generate_final_recommendation()

    # Save detailed report
    print("\n" + "="*80)
    print("Saving detailed comparison report...")
    print("="*80)

    with open('doc2vec_comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DOC2VEC CONFIGURATION COMPARISON - DETAILED REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("QUANTITATIVE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("RECOMMENDED CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Best: {best_config}\n")
        f.write(f"Composite Score: {scores['composite_score']:.4f}\n")
        f.write(f"Number of Clusters: {scores['n_clusters']}\n\n")

        f.write("See doc2vec_comparison_metrics.png for visualizations\n")

    print("\nReport saved to: doc2vec_comparison_report.txt")
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
