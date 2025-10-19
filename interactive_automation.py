#!/usr/bin/env python3
"""
Interactive Automation Script for Lab 5
- Runs periodic data collection at specified intervals
- Between updates, provides interactive cluster search
- User can input keywords/messages to find matching clusters
- Displays cluster results with graphical representation
"""

import time
import argparse
import threading
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from reddit_data_processor import RedditDataProcessor
from database_connection import SQLiteConnection

VISUALIZATION_DIR = Path("visualizations")

class InteractiveAutomation:
    def __init__(self, client_id, client_secret, user_agent, subreddit, posts_per_run,
                 interval_minutes, data_dir="reddit_data"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.subreddit = subreddit
        self.posts_per_run = posts_per_run
        self.interval_minutes = interval_minutes
        self.data_dir = data_dir

        self.running = True
        self.updating = False
        self.last_update = None
        # First run starts immediately, next update will be interval_minutes after that
        self.next_update = datetime.now() + timedelta(minutes=interval_minutes)

        # Load Doc2Vec model if available
        self.doc2vec_model = None
        self.load_latest_model()

        print("\n" + "="*80)
        print("[AUTOMATION] INTERACTIVE AUTOMATION STARTED")
        print("="*80)
        print(f"Subreddit: r/{subreddit}")
        print(f"Posts per run: {posts_per_run}")
        print(f"Update interval: {interval_minutes} minutes")
        print(f"Database: SQLite (reddit_data)")
        print("="*80 + "\n")

    def load_latest_model(self):
        """Load the most recent Doc2Vec model"""
        model_dir = Path(self.data_dir) / "embeddings"
        if not model_dir.exists():
            return

        model_files = sorted(model_dir.glob("*_doc2vec.model"))
        if model_files:
            latest_model = model_files[-1]
            try:
                self.doc2vec_model = Doc2Vec.load(str(latest_model))
                print(f"[SUCCESS] Loaded Doc2Vec model: {latest_model.name}")
            except Exception as e:
                print(f"[WARNING]  Could not load Doc2Vec model: {e}")

    def run_data_collection(self):
        """Run one cycle of data collection"""
        self.updating = True
        print("\n" + "="*80)
        print(f"[CYCLE] DATA COLLECTION CYCLE STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        try:
            print(f"[FETCHING] Fetching data from r/{self.subreddit}...")
            processor = RedditDataProcessor(
                self.client_id,
                self.client_secret,
                self.user_agent,
                self.data_dir
            )

            try:
                print(f"[PROCESSING]  Processing data...")
                result = processor.process_data(
                    self.subreddit,
                    self.posts_per_run,
                    'all',
                    recluster_all=True
                )

                if result:
                    print(f"[SUCCESS] Successfully processed {len(result['data'])} posts")
                    print(f"[DATABASE] Database updated: {result['database']}")
                    print(f"[TIME]  Processing time: {result['processing_time']:.1f} seconds")

                    # Reload model with new data
                    self.load_latest_model()

                    self.last_update = datetime.now()
                    self.next_update = self.last_update + timedelta(minutes=self.interval_minutes)
                else:
                    print("[ERROR] Data collection failed")

            finally:
                processor.close()

        except Exception as e:
            print(f"[ERROR] ERROR during data collection: {e}")

        self.updating = False

        # Always update next_update time, even if there was an error
        if self.last_update is None:
            self.last_update = datetime.now()
        self.next_update = datetime.now() + timedelta(minutes=self.interval_minutes)

        print("="*80)
        print(f"[SUCCESS] DATA COLLECTION CYCLE COMPLETE")
        if self.next_update:
            print(f"[NEXT] Next update: {self.next_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")

    def find_closest_cluster(self, query_text):
        """Find cluster closest to user's query"""
        if not self.doc2vec_model:
            print("[ERROR] Doc2Vec model not available. Run data collection first.")
            return None

        db_conn = None
        try:
            # Generate embedding for query
            tokens = simple_preprocess(query_text, deacc=True)
            if not tokens:
                print("[ERROR] Invalid query - no meaningful words found")
                return None

            query_vector = self.doc2vec_model.infer_vector(tokens)

            # Load cluster centroids and find closest
            db_conn = SQLiteConnection()

            # Get all posts with embeddings and cluster assignments
            results = db_conn.execute_query("""
                SELECT c.cluster_id, p.embedding
                FROM clusters c
                JOIN posts p ON c.post_id = p.id
                WHERE p.embedding IS NOT NULL
            """, fetch='all')

            if not results:
                print("[ERROR] No clusters found in database")
                return None

            cluster_vectors = {}
            for cluster_id, embedding_json in results:
                try:
                    embedding = np.array(json.loads(embedding_json))
                    if cluster_id not in cluster_vectors:
                        cluster_vectors[cluster_id] = []
                    cluster_vectors[cluster_id].append(embedding)
                except:
                    continue

            if not cluster_vectors:
                print("[ERROR] No valid embeddings found")
                return None

            # Calculate cluster centroids
            centroids = {}
            for cluster_id, vectors in cluster_vectors.items():
                centroids[cluster_id] = np.mean(vectors, axis=0)

            # Find closest cluster using cosine similarity
            max_similarity = -1
            closest_cluster = None
            best_distance = 0
            all_similarities = {}

            for cluster_id, centroid in centroids.items():
                # Cosine similarity (higher is better, range -1 to 1)
                similarity = cosine_similarity([query_vector], [centroid])[0][0]
                all_similarities[cluster_id] = similarity
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_cluster = cluster_id
                    # Convert to distance for display (lower is better)
                    best_distance = 1 - similarity

            # Show similarity to all clusters for debugging
            print(f"[DEBUG] Similarities to all clusters:")
            for cid in sorted(all_similarities.keys()):
                print(f"   Cluster {cid}: {all_similarities[cid]:.4f}")

            return closest_cluster, best_distance, query_vector

        except Exception as e:
            print(f"[ERROR] Error finding cluster: {e}")
            return None
        finally:
            if db_conn:
                db_conn.close()

    def display_cluster_info(self, cluster_id, query_vector=None):
        """Display detailed info about a cluster with visualization"""
        db_conn = SQLiteConnection()

        # Get cluster posts with embeddings
        results = db_conn.execute_query("""
            SELECT p.id, p.title, p.cleaned_content, p.keywords, p.embedding, c.distance
            FROM clusters c
            JOIN posts p ON c.post_id = p.id
            WHERE c.cluster_id = ?
        """, (cluster_id,), fetch='all')

        posts = []
        all_keywords = []

        for post_id, title, content, keywords_json, embedding_json, distance in results:
            try:
                keywords = json.loads(keywords_json) if keywords_json else []
            except:
                keywords = []

            try:
                embedding = np.array(json.loads(embedding_json)) if embedding_json else None
            except:
                embedding = None

            posts.append({
                'id': post_id,
                'title': title or '',
                'content': content or '',
                'keywords': keywords,
                'embedding': embedding,
                'distance': distance
            })
            all_keywords.extend(keywords)

        # If query_vector provided, sort by similarity to query instead of distance to centroid
        if query_vector is not None:
            for post in posts:
                if post['embedding'] is not None:
                    similarity = cosine_similarity([query_vector], [post['embedding']])[0][0]
                    post['query_similarity'] = similarity
                    post['query_distance'] = 1 - similarity
                else:
                    post['query_similarity'] = -1
                    post['query_distance'] = 999

            # Sort by query similarity (highest first)
            posts.sort(key=lambda x: x['query_similarity'], reverse=True)

        # Limit to top 10
        posts = posts[:10]

        # Get cluster size
        cluster_size_result = db_conn.execute_query("""
            SELECT COUNT(*) FROM clusters WHERE cluster_id = ?
        """, (cluster_id,), fetch='one')
        cluster_size = cluster_size_result[0]

        db_conn.close()

        # Display cluster info
        print("\n" + "="*80)
        print(f"[DATA] CLUSTER {cluster_id} DETAILS")
        print("="*80)
        print(f"Total posts in cluster: {cluster_size}")

        # Top keywords
        if all_keywords:
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(10)
            print(f"\n[KEYWORDS] Top Keywords:")
            for keyword, count in top_keywords:
                print(f"   • {keyword}: {count}")

        # Representative posts
        if query_vector is not None:
            print(f"\n[POSTS] Top 5 Posts Most Similar to Your Query:")
            for i, post in enumerate(posts[:5], 1):
                print(f"\n   {i}. {post['title']}")
                print(f"      Similarity to query: {post['query_similarity']:.3f} (distance: {post['query_distance']:.3f})")
                if post['content']:
                    content_preview = post['content'][:150] + "..." if len(post['content']) > 150 else post['content']
                    print(f"      Content: {content_preview}")
        else:
            print(f"\n[POSTS] Top 5 Representative Posts (closest to centroid):")
            for i, post in enumerate(posts[:5], 1):
                print(f"\n   {i}. {post['title']}")
                print(f"      Distance to centroid: {post['distance']:.3f}")
                if post['content']:
                    content_preview = post['content'][:150] + "..." if len(post['content']) > 150 else post['content']
                    print(f"      Content: {content_preview}")

        print("="*80 + "\n")

        # Create visualization
        self.create_cluster_visualization(cluster_id, posts, all_keywords, cluster_size)

    def create_cluster_visualization(self, cluster_id, posts, all_keywords, cluster_size):
        """Create and display a visualization for the cluster"""
        VISUALIZATION_DIR.mkdir(exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Top keywords bar chart
        if all_keywords:
            keyword_counts = Counter(all_keywords)
            top_keywords = keyword_counts.most_common(8)
            words = [kw for kw, _ in top_keywords]
            counts = [cnt for _, cnt in top_keywords]

            colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
            ax1.barh(range(len(words)), counts, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_yticks(range(len(words)))
            ax1.set_yticklabels(words)
            ax1.invert_yaxis()
            ax1.set_xlabel('Frequency', fontweight='bold')
            ax1.set_title(f'Cluster {cluster_id} - Top Keywords', fontweight='bold', fontsize=14)
            ax1.grid(axis='x', alpha=0.3)

            # Add count labels
            for i, count in enumerate(counts):
                ax1.text(count, i, f' {count}', va='center', fontweight='bold')

        # Right: Post similarities/distances
        if posts:
            # Check if we have query similarity scores
            has_query_sim = 'query_similarity' in posts[0]

            if has_query_sim:
                # Show cosine similarity to query (higher is better)
                similarities = [p['query_similarity'] for p in posts[:10]]
                colors = plt.cm.RdYlGn(np.linspace(0, 1, len(similarities)))
                ax2.barh(range(len(similarities)), similarities, color=colors, alpha=0.8, edgecolor='black')
                ax2.set_yticks(range(len(similarities)))
                ax2.set_yticklabels([f"Post {i+1}" for i in range(len(similarities))], fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel('Cosine Similarity to Query', fontweight='bold')
                ax2.set_title(f'Cluster {cluster_id} - Posts Most Similar to Query', fontweight='bold', fontsize=14)
                ax2.grid(axis='x', alpha=0.3)

                # Add similarity labels
                for i, sim in enumerate(similarities):
                    ax2.text(sim, i, f' {sim:.3f}', va='center', fontsize=8, fontweight='bold')
            else:
                # Show distance to centroid (lower is better)
                distances = [p['distance'] for p in posts[:10]]
                colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(distances)))
                ax2.barh(range(len(distances)), distances, color=colors, alpha=0.8, edgecolor='black')
                ax2.set_yticks(range(len(distances)))
                ax2.set_yticklabels([f"Post {i+1}" for i in range(len(distances))], fontsize=9)
                ax2.invert_yaxis()
                ax2.set_xlabel('Distance from Centroid', fontweight='bold')
                ax2.set_title(f'Cluster {cluster_id} - Post Similarity', fontweight='bold', fontsize=14)
                ax2.grid(axis='x', alpha=0.3)

                # Add distance labels
                for i, dist in enumerate(distances):
                    ax2.text(dist, i, f' {dist:.3f}', va='center', fontsize=8, fontweight='bold')

        plt.suptitle(f'Cluster {cluster_id} Analysis ({cluster_size} posts total)',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_file = VISUALIZATION_DIR / f'cluster_{cluster_id}_search_result.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[DATA] Visualization saved: {output_file}")
        print(f"[TIP] To view: open {output_file}")

    def interactive_prompt(self):
        """Interactive prompt for cluster search"""
        print("\n" + "="*80)
        print("[SEARCH] INTERACTIVE CLUSTER SEARCH")
        print("="*80)
        print("Enter keywords or a message to find matching clusters.")
        print("Commands: 'exit' or 'quit' to stop, 'status' for update info")
        print("="*80 + "\n")

        while self.running:
            try:
                if self.updating:
                    time.sleep(1)
                    continue

                user_input = input("[SEARCH] Search query (or command): ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    self.running = False
                    break

                if user_input.lower() == 'status':
                    print(f"\n[DATA] System Status:")
                    print(f"   Last update: {self.last_update.strftime('%H:%M:%S') if self.last_update else 'Never'}")
                    print(f"   Next update: {self.next_update.strftime('%H:%M:%S') if self.next_update else 'Soon'}")
                    print(f"   Currently updating: {'Yes' if self.updating else 'No'}")
                    continue

                print(f"\n[SEARCH] Searching for: '{user_input}'...")
                result = self.find_closest_cluster(user_input)

                if result:
                    cluster_id, distance, query_vector = result
                    print(f"[SUCCESS] Found closest match: Cluster {cluster_id} (distance: {distance:.3f})")
                    self.display_cluster_info(cluster_id, query_vector)

            except KeyboardInterrupt:
                print("\n\n Exiting...")
                self.running = False
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")

    def start_automation(self):
        """Start the automation loop in a separate thread"""
        def automation_loop():
            # Run first collection immediately
            self.run_data_collection()

            while self.running:
                time.sleep(60)  # Check every minute

                if self.next_update and datetime.now() >= self.next_update:
                    self.run_data_collection()

        # Start automation thread
        automation_thread = threading.Thread(target=automation_loop, daemon=True)
        automation_thread.start()

        # Start interactive prompt in main thread
        self.interactive_prompt()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Automation - Periodic data collection with cluster search",
        epilog="Example: python interactive_automation.py 5"
    )
    parser.add_argument('interval', type=int, nargs='?', default=60,
                       help='Update interval in minutes (e.g., 5 for every 5 minutes)')
    parser.add_argument('--subreddit', default='iphone', help='Subreddit to scrape')
    parser.add_argument('--posts', type=int, default=100, help='Posts to fetch per update')
    parser.add_argument('--client-id', default='R4r2pV4_CLRBpr_Csx_F_A')
    parser.add_argument('--client-secret', default='ZHMtPtLx-PSYMAriIOXG4hD6XgkqlA')
    parser.add_argument('--user-agent', default='Beginning-Split862')

    args = parser.parse_args()

    # Create and start interactive automation
    automation = InteractiveAutomation(
        args.client_id,
        args.client_secret,
        args.user_agent,
        args.subreddit,
        args.posts,
        args.interval
    )

    automation.start_automation()


if __name__ == '__main__':
    main()