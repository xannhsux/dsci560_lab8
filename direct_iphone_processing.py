#!/usr/bin/env python3
"""
Direct iPhone data processing - no command line needed
Just run this file directly: python direct_iphone_processing.py
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our processor
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from reddit_data_processor import RedditDataProcessor

def process_iphone_data():
    """Process iPhone subreddit data with predefined settings"""
    
    # Reddit API credentials
    CLIENT_ID = "R4r2pV4_CLRBpr_Csx_F_A"
    CLIENT_SECRET = "ZHMtPtLx-PSYMAriIOXG4hD6XgkqlA"
    USER_AGENT = "Beginning-Split862"
    
    # Settings for iPhone data processing
    config = {
        'subreddit': 'iphone',
        'num_posts': 5000,
        'data_dir': 'reddit_data',
        'save_format': 'all'  # Save in all formats
    }
    
    print("=" * 60)
    print("IPHONE REDDIT DATA PROCESSING")
    print("=" * 60)
    print(f"Subreddit: r/{config['subreddit']}")
    print(f"Target posts: {config['num_posts']}")
    print(f"Data directory: {config['data_dir']}")
    print(f"Save formats: {config['save_format']}")
    print("=" * 60)
    
    try:
        # Initialize the processor
        processor = RedditDataProcessor(
            CLIENT_ID, 
            CLIENT_SECRET, 
            USER_AGENT, 
            config['data_dir']
        )
        
        # Run the processing
        result = processor.process_data(
            config['subreddit'], 
            config['num_posts'], 
            config['save_format']
        )
        
        if result:
            print("\n" + "=" * 60)
            print("[SUCCESS] PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Session ID: {result['session_id']}")
            print(f"Posts processed: {len(result['data'])}")
            print(f"Processing time: {result['processing_time']:.1f} seconds")
            print(f"Average time per post: {result['processing_time']/len(result['data']):.2f} seconds")

            print(f"\nFiles created:")
            for format_type, filepath in result['files'].items():
                file_size = os.path.getsize(filepath) / 1024  # Size in KB
                print(f"  {format_type.upper():12}: {filepath} ({file_size:.1f} KB)")

            # Handle both SQLite (database_path) and MySQL (database) formats
            if 'database_path' in result:
                print(f"  DATABASE   : {result['database_path']}")
            elif 'database' in result:
                print(f"  DATABASE   : MySQL ({result['database']})")

            print(f"\nData ready for:")
            print("  • Clustering algorithms (embeddings folder)")
            print("  • Automation scripts (processed folder)")
            print("  • Manual analysis (JSON files)")

            # Show some quick stats
            posts_data = result['data']
            avg_score = sum(post['features']['score'] for post in posts_data) / len(posts_data)
            posts_with_images = sum(1 for post in posts_data if post['features']['has_image'])

            print(f"\nQuick Stats:")
            print(f"  • Average post score: {avg_score:.1f}")
            print(f"  • Posts with images: {posts_with_images} ({posts_with_images/len(posts_data)*100:.1f}%)")
            if result.get('embedding_feature_names'):
                print(f"  • Embedding dimensions: {len(result['embedding_feature_names'])}")
            if result.get('doc2vec_vector_size'):
                print(f"  • Doc2Vec vector size: {result['doc2vec_vector_size']}")
            if result.get('clusters') and result['clusters'].get('centroid_examples'):
                print(f"  • Clusters generated: {result['clusters']['n_clusters']}")
                print("  • Centroid exemplars:")
                for example in result['clusters']['centroid_examples']:
                    title_or_text = example['title'] or example['cleaned_content']
                    preview = (title_or_text[:80] + '...') if len(title_or_text) > 80 else title_or_text
                    print(f"    - Cluster {example['cluster_id']}: {preview} (distance {example['distance']:.3f})")

            return result
        else:
            print("[FAILED] Processing failed - check logs for details")
            return None

    except Exception as e:
        print(f"[ERROR] Error during processing: {e}")
        return None

def main():
    """Main function - just run the iPhone processing"""
    result = process_iphone_data()

    if result:
        print(f"\nAll done! Check the 'reddit_data' folder for your processed iPhone data.")
        print(f"Next step: Use the clustering-ready data in 'reddit_data/embeddings/' folder")
    else:
        print(f"\n[ERROR] Something went wrong. Check the error messages above.")

if __name__ == "__main__":
    main()
