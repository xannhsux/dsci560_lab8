#!/usr/bin/env python3
"""Recluster all posts in the database with updated formula"""

from reddit_data_processor import RedditDataProcessor

CLIENT_ID = "R4r2pV4_CLRBpr_Csx_F_A"
CLIENT_SECRET = "ZHMtPtLx-PSYMAriIOXG4hD6XgkqlA"
USER_AGENT = "Beginning-Split862"

print("=" * 70)
print("RECLUSTERING ALL POSTS")
print("=" * 70)

processor = RedditDataProcessor(CLIENT_ID, CLIENT_SECRET, USER_AGENT, "reddit_data")

print("\nLoading all posts from database...")
all_posts = processor.load_existing_posts_from_database()
print(f"Loaded {len(all_posts)} posts with embeddings")

if len(all_posts) > 0:
    print(f"\nTraining Doc2Vec model on all {len(all_posts)} posts...")
    doc2vec_result = processor.generate_doc2vec_embeddings(all_posts, vector_size=100, epochs=40)

    if doc2vec_result:
        print(f"[SUCCESS] Trained Doc2Vec model with {len(doc2vec_result['vectors'])} vectors")

        # Update embeddings in posts
        vectors_by_id = {doc_id: vector for doc_id, vector in zip(doc2vec_result['ids'], doc2vec_result['vectors'])}
        for post in all_posts:
            post['embedding'] = vectors_by_id.get(post['id'])

        print(f"\nReclustering with new formula (4-8 clusters)...")
        print(f"Expected clusters for {len(all_posts)} posts: {max(4, min(8, len(all_posts) // 60))}")

        clustering_result = processor.cluster_posts(all_posts)

        if clustering_result:
            print(f"\n[SUCCESS] Clustered into {clustering_result['n_clusters']} clusters")

            # Delete old cluster assignments
            print("\nDeleting old cluster assignments...")
            processor.delete_all_cluster_assignments()

            # Save new cluster assignments
            print(f"Saving {len(clustering_result['assignments'])} new cluster assignments...")
            processor.save_clusters_to_database(clustering_result['assignments'], processor.session_id)

            # Save the new Doc2Vec model
            from pathlib import Path
            model_path = Path('reddit_data/embeddings') / f"iphone_{processor.session_id}_doc2vec.model"
            doc2vec_result['model'].save(str(model_path))
            print(f"\n[SUCCESS] Saved Doc2Vec model: {model_path}")

        print("\n[SUCCESS] Reclustering complete!")
        print("\nCluster distribution:")

        # Count posts per cluster
        cluster_counts = {}
        for assignment in clustering_result['assignments']:
            cluster_id = assignment['cluster_id']
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        for cluster_id in sorted(cluster_counts.keys()):
            print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} posts")

        print("\nCentroid exemplars:")
        for example in clustering_result['centroid_examples']:
            title = example['title'][:60] + "..." if len(example['title']) > 60 else example['title']
            print(f"  Cluster {example['cluster_id']}: {title}")
    else:
        print("[ERROR] Clustering failed")
else:
    print("[ERROR] No posts found to cluster")

processor.close()
print("\n" + "=" * 70)
print("Done! You can now use interactive_automation.py")