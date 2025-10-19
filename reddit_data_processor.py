import praw
import json
import re
import html
import time
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob
import hashlib
import pytesseract
from PIL import Image
import io
import os
import argparse
import logging
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from database_connection import SQLiteConnection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

from io import BytesIO


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedditDataProcessor:
    def __init__(self, client_id, client_secret, user_agent, data_dir="reddit_data"):
        """Initialize Reddit API connection and data storage system"""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Create organized directory structure for different data types
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw_data"           # Original scraped data
        self.processed_dir = self.data_dir / "processed"     # Cleaned and processed data
        self.embeddings_dir = self.data_dir / "embeddings"  # For clustering algorithms
        self.metadata_dir = self.data_dir / "metadata"      # Index and tracking files

        # Create all necessary directories
        for directory in [self.data_dir, self.raw_dir, self.processed_dir, 
                         self.embeddings_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)

        # Initialize MySQL database connection for processed posts
        self.db_conn = SQLiteConnection()
        self._initialize_database()

        # Initialize data tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"Data processing session started: {self.session_id}")
        logging.info(f"Data storage initialized at: {self.data_dir}")

    def load_existing_posts_from_database(self):
        """Load all existing posts from database with their embeddings"""
        try:
            results = self.db_conn.execute_query("""
                SELECT id, title, content, cleaned_content, embedding, keywords, topics,
                       subreddit, features, created_datetime
                FROM posts
                WHERE embedding IS NOT NULL
            """, fetch='all')

            posts = []
            for row in results:
                post_id, title, content, cleaned_content, embedding_json, keywords_json, topics_json, \
                subreddit, features_json, created_datetime = row

                # Parse JSON fields
                try:
                    embedding = json.loads(embedding_json) if embedding_json else None
                    keywords = json.loads(keywords_json) if keywords_json else []
                    topics = json.loads(topics_json) if topics_json else []
                    features = json.loads(features_json) if features_json else {}
                except:
                    embedding = None
                    keywords = []
                    topics = []
                    features = {}

                if embedding:  # Only include posts with embeddings
                    posts.append({
                        'id': post_id,
                        'title': title or '',
                        'content': content or '',
                        'cleaned_content': cleaned_content or '',
                        'embedding': embedding,
                        'keywords': keywords,
                        'topics': topics,
                        'subreddit': subreddit,
                        'features': features  # Use features as-is, already has score, num_comments, etc.
                    })

            logging.info(f"Loaded {len(posts)} existing posts from database")
            return posts
        except Exception as e:
            logging.error(f"Error loading posts from database: {e}")
            return []

    def delete_all_cluster_assignments(self):
        """Delete all cluster assignments to prepare for reclustering"""
        try:
            self.db_conn.execute_query("DELETE FROM clusters", fetch=None)
            logging.info("Deleted all existing cluster assignments")
        except Exception as e:
            logging.error(f"Error deleting cluster assignments: {e}")

    def merge_posts(self, existing_posts, new_posts):
        """Merge new posts with existing, avoiding duplicates"""
        existing_ids = {post['id'] for post in existing_posts}
        merged = list(existing_posts)  # Start with all existing

        new_count = 0
        for post in new_posts:
            if post['id'] not in existing_ids:
                merged.append(post)
                new_count += 1

        logging.info(f"Merged posts: {len(existing_posts)} existing + {new_count} new = {len(merged)} total")
        return merged

    def close(self):
        """Release database resources when finished."""
        try:
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
        except Exception as db_error:
            logging.warning(f"Error closing database connection: {db_error}")

    def __del__(self):
        self.close()
    
    def hash_username(self, username):
        """Hash usernames for privacy protection"""
        if username is None or username == '[deleted]':
            return 'anonymous'
        return hashlib.sha256(username.encode()).hexdigest()[:16]  # Shortened hash for efficiency
    
    def clean_text(self, text):
        """Advanced text cleaning and preprocessing"""
        if not text:
            return ""
        
        # Remove HTML tags and decode entities
        text = html.unescape(text)
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove URLs, mentions, and hashtags but keep them for analysis
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        mention_pattern = r'@\w+'
        hashtag_pattern = r'#\w+'
        
        # Extract these for metadata
        urls = re.findall(url_pattern, text)
        mentions = re.findall(mention_pattern, text)
        hashtags = re.findall(hashtag_pattern, text)
        
        # Clean the main text
        text = re.sub(url_pattern, ' [URL] ', text)
        text = re.sub(mention_pattern, ' [USER] ', text)
        text = re.sub(hashtag_pattern, ' [TAG] ', text)
        
        # Remove special characters but preserve sentence structure
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"@#$%&*+=\[\]]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove promotional content patterns
        promotional_patterns = [
            r'\[promoted\].*?\[/promoted\]',
            r'sponsored by.*?(?:\n|\.)',
            r'advertisement.*?(?:\n|\.)',
            r'buy now.*?(?:\n|\.)',
            r'click here.*?(?:\n|\.)',
            r'visit our website.*?(?:\n|\.)',
            r'subscribe.*?(?:\n|\.)',
            r'follow us.*?(?:\n|\.)'
        ]
        
        for pattern in promotional_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return {
            'cleaned_text': text.strip(),
            'extracted_urls': urls,
            'extracted_mentions': mentions,
            'extracted_hashtags': hashtags
        }

    def _initialize_database(self):
        """Create required database tables if they do not exist."""
        create_posts_sql = """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            subreddit TEXT,
            title TEXT,
            content TEXT,
            cleaned_content TEXT,
            image_text TEXT,
            keywords TEXT,
            topics TEXT,
            extracted_urls TEXT,
            extracted_mentions TEXT,
            extracted_hashtags TEXT,
            features TEXT,
            embedding TEXT,
            created_datetime TEXT,
            processed_timestamp TEXT
        );
        """
        create_clusters_sql = """
        CREATE TABLE IF NOT EXISTS clusters (
            post_id TEXT PRIMARY KEY,
            session_id TEXT,
            cluster_id INTEGER,
            distance REAL,
            FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE
        );
        """
        self.db_conn.execute_query(create_posts_sql)
        self.db_conn.execute_query(create_clusters_sql)
    
    def extract_image_text(self, url):
        """Extract text from images using OCR with error handling"""
        try:
            if not url or not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                return ""
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, timeout=15, headers=headers)
            
            if response.status_code == 200 and len(response.content) > 0:
                try:
                    image = Image.open(io.BytesIO(response.content))
                    # Convert to RGB if necessary
                    if image.mode not in ('RGB', 'L'):
                        image = image.convert('RGB')
                    
                    # Extract text using Tesseract
                    extracted_text = pytesseract.image_to_string(image, config='--psm 6')
                    
                    if extracted_text.strip():
                        cleaned_result = self.clean_text(extracted_text)
                        return cleaned_result['cleaned_text']
                        
                except Exception as img_error:
                    logging.warning(f"Image processing failed for {url}: {img_error}")
                    
        except Exception as e:
            logging.warning(f"Failed to extract text from image {url}: {e}")
        
        return ""

    def save_posts_to_database(self, posts_data):
        """Persist processed posts into the MySQL database."""
        if not posts_data:
            logging.warning("No posts available for database persistence")
            return 0

        rows_inserted = 0
        for post in posts_data:
            try:
                self.db_conn.execute_query(
                    """
                    INSERT OR REPLACE INTO posts (
                        id,
                        session_id,
                        subreddit,
                        title,
                        content,
                        cleaned_content,
                        image_text,
                        keywords,
                        topics,
                        extracted_urls,
                        extracted_mentions,
                        extracted_hashtags,
                        features,
                        embedding,
                        created_datetime,
                        processed_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        post.get('id'),
                        post.get('session_id'),
                        post.get('subreddit'),
                        post.get('title'),
                        post.get('content'),
                        post.get('cleaned_content'),
                        post.get('image_text'),
                        json.dumps(post.get('keywords', []), ensure_ascii=False),
                        json.dumps(post.get('topics', []), ensure_ascii=False),
                        json.dumps(post.get('extracted_urls', []), ensure_ascii=False),
                        json.dumps(post.get('extracted_mentions', []), ensure_ascii=False),
                        json.dumps(post.get('extracted_hashtags', []), ensure_ascii=False),
                        json.dumps(post.get('features', {}), ensure_ascii=False),
                        json.dumps(post.get('embedding'), ensure_ascii=False) if post.get('embedding') is not None else None,
                        post.get('created_datetime'),
                        post.get('processed_timestamp')
                    )
                )
                rows_inserted += 1
            except Exception as db_error:
                logging.error(f"Failed to persist post {post.get('id')}: {db_error}")

        logging.info(f"Persisted {rows_inserted} posts to SQLite database")
        return rows_inserted

    def save_clusters_to_database(self, cluster_assignments, session_id):
        """Persist cluster assignments into the database."""
        if not cluster_assignments:
            return 0

        for entry in cluster_assignments:
            try:
                self.db_conn.execute_query(
                    """
                    INSERT OR REPLACE INTO clusters (post_id, session_id, cluster_id, distance)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        entry['post_id'],
                        session_id,
                        entry['cluster_id'],
                        entry['distance']
                    )
                )
            except Exception as db_error:
                logging.error(f"Failed to persist cluster info for {entry['post_id']}: {db_error}")

        logging.info(f"Persisted {len(cluster_assignments)} cluster assignments")
        return len(cluster_assignments)

    def extract_keywords_and_topics(self, text, min_word_length=3, max_phrases=15):
        """Advanced keyword and topic extraction optimized for clustering"""
        if not text or len(text.strip()) < 10:
            return [], [], {}

        try:
            blob = TextBlob(text)
            
            # Define comprehensive stop words
            stop_words = {
                'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'their', 
                'would', 'could', 'should', 'there', 'where', 'when', 'what', 'which',
                'about', 'after', 'again', 'before', 'being', 'below', 'between', 'both',
                'during', 'each', 'more', 'most', 'other', 'some', 'such', 'than',
                'very', 'were', 'while', 'here', 'just', 'now', 'only', 'own', 'same',
                'those', 'through', 'under', 'until', 'up', 'down', 'out', 'off', 'over',
                'then', 'once', 'how', 'why', 'does', 'did', 'doing', 'reddit', 'post',
                'comment', 'thread', 'subreddit', '[url]', '[user]', '[tag]'
            }
            
            # Extract noun phrases as potential keywords
            keywords = []
            for phrase in blob.noun_phrases:
                phrase_clean = phrase.lower().strip()
                if (len(phrase_clean) >= min_word_length and 
                    len(phrase.split()) <= 4 and 
                    phrase_clean not in stop_words and
                    not any(stop in phrase_clean for stop in ['[url]', '[user]', '[tag]'])):
                    keywords.append(phrase_clean)
            
            # Extract individual significant words
            word_freq = defaultdict(int)
            for word in blob.words:
                word_lower = word.lower().strip()
                if (len(word_lower) >= min_word_length and 
                    word_lower not in stop_words and
                    word_lower.isalpha()):  # Only alphabetic words
                    word_freq[word_lower] += 1
            
            # Get top words as topics
            topics = [word for word, freq in sorted(word_freq.items(), 
                     key=lambda x: x[1], reverse=True) if freq > 1][:10]
            
            # Calculate text statistics for clustering features
            text_stats = {
                'char_count': len(text),
                'word_count': len(blob.words),
                'sentence_count': len(blob.sentences),
                'avg_word_length': np.mean([len(word) for word in blob.words]) if blob.words else 0,
                'sentiment_polarity': blob.sentiment.polarity,
                'sentiment_subjectivity': blob.sentiment.subjectivity,
                'keyword_density': len(keywords) / len(blob.words) if blob.words else 0,
                'unique_word_ratio': len(set(blob.words)) / len(blob.words) if blob.words else 0
            }
            
            return keywords[:max_phrases], topics, text_stats
            
        except Exception as e:
            logging.warning(f"Failed to extract keywords and topics: {e}")
            return [], [], {}
    
    def process_single_post(self, post, subreddit_name):
        """Process individual post with comprehensive feature extraction"""
        try:
            post_id = post.id
            
            # Extract image text if available
            image_text = ""
            image_urls = []
            # Disable slow image processing for performance
            # if hasattr(post, 'url') and post.url:
            #     image_text = self.extract_image_text(post.url)
            #     if image_text:
            #         image_urls.append(post.url)
            
            # Combine all textual content
            title_text = post.title or ""
            content_text = post.selftext or ""
            combined_text = f"{title_text} {content_text} {image_text}".strip()
            
            # Clean and process text
            cleaning_result = self.clean_text(combined_text)
            cleaned_content = cleaning_result['cleaned_text']
            
            # Extract keywords, topics, and text statistics
            keywords, topics, text_stats = self.extract_keywords_and_topics(cleaned_content)
            
            # Create comprehensive post data structure optimized for clustering
            post_data = {
                # Basic identifiers
                'id': post_id,
                'subreddit': subreddit_name,
                
                # Original content
                'title': title_text,
                'content': content_text,
                'url': post.url,
                
                # Processed content
                'cleaned_content': cleaned_content,
                'image_text': image_text,
                
                # Extracted features for clustering
                'keywords': keywords,
                'topics': topics,
                'extracted_urls': cleaning_result['extracted_urls'],
                'extracted_mentions': cleaning_result['extracted_mentions'],
                'extracted_hashtags': cleaning_result['extracted_hashtags'],
                
                # Numerical features for clustering algorithms
                'features': {
                    'score': int(post.score),
                    'num_comments': int(post.num_comments),
                    'created_utc': int(post.created_utc),
                    'text_length': len(cleaned_content),
                    'title_length': len(title_text),
                    'has_image': bool(image_text),
                    'url_count': len(cleaning_result['extracted_urls']),
                    'mention_count': len(cleaning_result['extracted_mentions']),
                    'hashtag_count': len(cleaning_result['extracted_hashtags']),
                    **text_stats  # Include all text statistics
                },
                
                # Metadata
                'author_hash': self.hash_username(str(post.author) if post.author else None),
                'created_datetime': datetime.fromtimestamp(post.created_utc).isoformat(),
                'processed_timestamp': datetime.now().isoformat(),
                'session_id': self.session_id
            }
            
            return post_data
            
        except Exception as e:
            logging.error(f"Error processing post {post.id}: {e}")
            return None

    def generate_tfidf_embeddings(self, posts_data, max_features=300):
        """Generate TF-IDF embeddings for optional vector-based analytics."""
        documents = [f"{post.get('title', '')} {post.get('cleaned_content', '')}".strip()
                     for post in posts_data]

        non_empty_docs = any(doc for doc in documents)
        if not posts_data or not non_empty_docs:
            logging.warning("Skipping embedding generation: no textual content available")
            return None

        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words='english',
                strip_accents='unicode'
            )
            matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out().tolist()
            embeddings = matrix.toarray().tolist()
            logging.info(f"Generated TF-IDF embeddings with {len(feature_names)} features")
            return {
                'vectors': embeddings,
                'feature_names': feature_names,
                'vectorizer': vectorizer
            }
        except Exception as embed_error:
            logging.error(f"Embedding generation failed: {embed_error}")
            return None

    def generate_doc2vec_embeddings(self, posts_data, vector_size=100, min_count=1, epochs=40):
        """Train a Doc2Vec model and produce dense embeddings for each post."""
        documents = []
        for post in posts_data:
            text_content = f"{post.get('title', '')} {post.get('cleaned_content', '')}".strip()
            tokens = simple_preprocess(text_content, deacc=True)
            if not tokens:
                continue
            documents.append(TaggedDocument(words=tokens, tags=[post['id']]))

        if not documents:
            logging.warning("Skipping Doc2Vec embedding: no documents contained tokenizable text")
            return None

        try:
            model = Doc2Vec(
                vector_size=vector_size,
                min_count=min_count,
                epochs=epochs,
                dm=1,
                workers=1
            )
            model.build_vocab(documents)
            model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

            embeddings = []
            doc_ids = []
            for doc in documents:
                vector = np.nan_to_num(np.array(model.dv[doc.tags[0]], dtype=float)).tolist()
                embeddings.append(vector)
                doc_ids.append(doc.tags[0])

            logging.info(f"Generated Doc2Vec embeddings with dimension {vector_size}")
            return {
                'vectors': embeddings,
                'ids': doc_ids,
                'model': model
            }
        except Exception as doc2vec_error:
            logging.error(f"Doc2Vec embedding failed: {doc2vec_error}")
            return None

    def fetch_posts_batch(self, subreddit_name, limit=100, timeout=180):
        """Fetch posts with comprehensive error handling and rate limiting"""
        posts_data = []
        start_time = time.time()

        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Try different sorting methods if one fails
            sorting_methods = ['hot', 'top', 'new']
            posts = None
            
            for method in sorting_methods:
                try:
                    if method == 'hot':
                        posts = subreddit.hot(limit=min(limit, 1000))
                    elif method == 'top':
                        posts = subreddit.top(limit=min(limit, 1000), time_filter='all')
                    elif method == 'new':
                        posts = subreddit.new(limit=min(limit, 1000))
                    break
                except Exception as sort_error:
                    logging.warning(f"Failed to fetch {method} posts: {sort_error}")
                    continue
            
            if not posts:
                logging.error("Failed to fetch posts with any sorting method")
                return []
            
            processed_count = 0
            failed_count = 0
            
            for post in posts:
                # Check timeout and limits
                if time.time() - start_time > timeout:
                    logging.warning(f"Timeout reached after {processed_count} posts")
                    break
                
                if len(posts_data) >= limit:
                    break
                
                # Process individual post
                post_data = self.process_single_post(post, subreddit_name)
                
                if post_data:
                    posts_data.append(post_data)
                    processed_count += 1
                else:
                    failed_count += 1
                
                # Progress logging and rate limiting
                if processed_count % 25 == 0:
                    logging.info(f"Processed {processed_count} posts, {failed_count} failed")
                    time.sleep(0.5)  # Brief pause to avoid rate limits
                
        except Exception as e:
            logging.error(f"Error in batch fetching: {e}")
        
        logging.info(f"Batch complete: {len(posts_data)} posts successfully processed")
        return posts_data
    
    def fetch_large_dataset(self, subreddit_name, total_posts, batch_timeout=180):
        """Handle large datasets with intelligent batching and retry logic"""
        all_posts = []
        batch_size = 400  # Smaller batches for long-running fetches
        retry_count = 0
        max_retries = 3
        
        while len(all_posts) < total_posts and retry_count < max_retries:
            remaining = total_posts - len(all_posts)
            current_batch_size = min(batch_size, remaining)
            
            logging.info(f"Fetching batch: {len(all_posts)}/{total_posts} (attempt {retry_count + 1})")
            
            batch_posts = self.fetch_posts_batch(
                subreddit_name, 
                current_batch_size, 
                batch_timeout
            )
            
            if not batch_posts:
                retry_count += 1
                if retry_count < max_retries:
                    logging.warning(f"Batch failed, retrying in 30 seconds... (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(30)
                continue
            
            # Reset retry counter on successful batch
            retry_count = 0
            all_posts.extend(batch_posts)
            
            # No wait between batches - optimized for speed
            if len(all_posts) < total_posts:
                logging.info(f"Processing next batch... ({len(all_posts)}/{total_posts} posts collected)")
        
        return all_posts[:total_posts]
    
    def save_processed_data(self, posts_data, subreddit_name, save_format='all',
                            embedding_feature_names=None, vectorizer=None,
                            doc2vec_vector_size=None):
        """Save data in multiple formats optimized for different use cases"""
        if not posts_data:
            logging.warning("No data to save")
            return {}

        timestamp = self.session_id
        base_filename = f"{subreddit_name}_{timestamp}"
        saved_files = {}
        
        try:
            # 1. Save raw processed data (JSON) - Human readable, good for debugging
            if save_format in ['json', 'all']:
                json_file = self.processed_dir / f"{base_filename}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(posts_data, f, ensure_ascii=False, indent=2)
                saved_files['json'] = json_file
                logging.info(f"Saved JSON data: {json_file}")
            
            # 2. Save binary data (Pickle) - Fast loading for automation
            if save_format in ['pickle', 'all']:
                pickle_file = self.processed_dir / f"{base_filename}.pkl"
                with open(pickle_file, 'wb') as f:
                    pickle.dump(posts_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_files['pickle'] = pickle_file
                logging.info(f"Saved pickle data: {pickle_file}")
            
            # 3. Save clustering-ready data structure
            if save_format in ['clustering', 'all']:
                clustering_data = self.prepare_clustering_data(posts_data, embedding_feature_names)
                clustering_file = self.embeddings_dir / f"{base_filename}_clustering.pkl"
                with open(clustering_file, 'wb') as f:
                    pickle.dump(clustering_data, f)
                saved_files['clustering'] = clustering_file
                logging.info(f"Saved clustering data: {clustering_file}")

                if vectorizer is not None:
                    vectorizer_file = self.embeddings_dir / f"{base_filename}_tfidf_vectorizer.pkl"
                    with open(vectorizer_file, 'wb') as f:
                        pickle.dump(vectorizer, f)
                    saved_files['vectorizer'] = vectorizer_file
                    logging.info(f"Saved TF-IDF vectorizer: {vectorizer_file}")

            # 4. Save metadata index for automation tracking
            metadata = {
                'session_id': self.session_id,
                'subreddit': subreddit_name,
                'total_posts': len(posts_data),
                'processing_date': datetime.now().isoformat(),
                'files': {k: str(v) for k, v in saved_files.items()},
                'post_ids': [post['id'] for post in posts_data],
                'date_range': {
                    'earliest': min(post['features']['created_utc'] for post in posts_data),
                    'latest': max(post['features']['created_utc'] for post in posts_data)
                },
                'embedding_feature_names': embedding_feature_names or [],
                'doc2vec_vector_size': doc2vec_vector_size
            }
            
            metadata_file = self.metadata_dir / f"{base_filename}_meta.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = metadata_file
            
        except Exception as e:
            logging.error(f"Error saving data: {e}")
        
        return saved_files
    
    def prepare_clustering_data(self, posts_data, embedding_feature_names=None):
        """Prepare data structure optimized for clustering algorithms"""
        clustering_ready = {
            'documents': [],      # Text content for vectorization
            'features': [],       # Numerical features
            'metadata': [],       # Post metadata
            'feature_names': [],  # Names of numerical features
            'embeddings': [],     # Precomputed document embeddings
            'embedding_feature_names': embedding_feature_names or [],
            'doc2vec_embeddings': []
        }
        
        # Get feature names from first post
        if posts_data:
            clustering_ready['feature_names'] = list(posts_data[0]['features'].keys())
        
        for post in posts_data:
            # Text for doc2vec/embedding
            text_content = f"{post['title']} {post['cleaned_content']}".strip()
            clustering_ready['documents'].append(text_content)
            
            # Numerical features for clustering
            feature_vector = [post['features'].get(name, 0) for name in clustering_ready['feature_names']]
            clustering_ready['features'].append(feature_vector)

            # Metadata for post identification
            clustering_ready['metadata'].append({
                'id': post['id'],
                'title': post['title'],
                'subreddit': post['subreddit'],
                'keywords': post['keywords'],
                'topics': post['topics']
            })

            # Precomputed embeddings
            if post.get('tfidf_vector') is not None:
                clustering_ready['embeddings'].append(post['tfidf_vector'])
            else:
                clustering_ready['embeddings'].append([])

            clustering_ready['doc2vec_embeddings'].append(post.get('embedding', []))

        return clustering_ready

    def _find_optimal_clusters(self, embeddings, min_k=3, max_k=10):
        """
        Use Elbow Method to find optimal number of clusters.
        Calculates WCSS (Within-Cluster Sum of Squares) for different k values
        and finds the elbow point using the rate of decrease.
        """
        n_samples = len(embeddings)

        # Adjust max_k based on dataset size
        # Rule of thumb: max clusters = sqrt(n/2)
        suggested_max = int(np.sqrt(n_samples / 2))
        max_k = min(max_k, max(min_k + 1, suggested_max))

        # Ensure we have enough samples
        max_k = min(max_k, n_samples - 1)

        if max_k < min_k:
            logging.warning(f"Not enough samples for clustering. Using {min_k} clusters.")
            return min_k

        wcss = []
        k_range = range(min_k, max_k + 1)

        logging.info(f"Finding optimal clusters using Elbow Method (testing k={min_k} to {max_k})...")

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                wcss.append(kmeans.inertia_)
            except Exception as e:
                logging.warning(f"Failed to compute k={k}: {e}")
                continue

        if len(wcss) < 2:
            logging.warning("Could not compute WCSS. Using default cluster count.")
            return min_k

        # Find elbow using rate of change (second derivative)
        # The elbow is where the rate of decrease slows down significantly
        deltas = np.diff(wcss)
        second_deltas = np.diff(deltas)

        if len(second_deltas) > 0:
            # Find the point with maximum second derivative (biggest change in slope)
            elbow_idx = np.argmax(second_deltas) + min_k
        else:
            # Fallback: use midpoint
            elbow_idx = min_k + len(wcss) // 2

        # Ensure we stay within valid range
        optimal_k = max(min_k, min(elbow_idx, max_k))

        logging.info(f"Optimal number of clusters determined: {optimal_k} (WCSS: {wcss})")

        return optimal_k

    def _merge_small_clusters(self, clustering_result, embeddings, index_map, posts_data, min_cluster_size=3):
        """
        Merge clusters with fewer than min_cluster_size posts into their nearest neighbor cluster.
        This handles outliers that form their own tiny clusters.
        """
        assignments = clustering_result['assignments']
        centroids = clustering_result['centroids']

        # Count posts per cluster
        cluster_counts = {}
        for assignment in assignments:
            cid = assignment['cluster_id']
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        # Find small clusters that need merging
        small_clusters = [cid for cid, count in cluster_counts.items() if count < min_cluster_size]

        if not small_clusters:
            logging.info("No small clusters to merge")
            return clustering_result

        logging.info(f"Found {len(small_clusters)} small clusters (< {min_cluster_size} posts): {small_clusters}")

        # For each small cluster, find nearest large cluster
        merge_map = {}  # small_cluster_id -> target_cluster_id
        large_clusters = [cid for cid, count in cluster_counts.items() if count >= min_cluster_size]

        if not large_clusters:
            # If all clusters are small, keep the original clustering
            logging.warning("All clusters are small. Keeping original clustering.")
            return clustering_result

        for small_cid in small_clusters:
            small_centroid = centroids[small_cid]

            # Find nearest large cluster by centroid distance
            min_distance = float('inf')
            nearest_cid = large_clusters[0]

            for large_cid in large_clusters:
                large_centroid = centroids[large_cid]
                distance = np.linalg.norm(small_centroid - large_centroid)

                if distance < min_distance:
                    min_distance = distance
                    nearest_cid = large_cid

            merge_map[small_cid] = nearest_cid
            logging.info(f"Merging cluster {small_cid} ({cluster_counts[small_cid]} posts) into cluster {nearest_cid} (distance: {min_distance:.3f})")

        # Reassign posts from small clusters
        new_assignments = []
        for assignment in assignments:
            old_cid = assignment['cluster_id']
            new_cid = merge_map.get(old_cid, old_cid)  # Use merge target if exists, otherwise keep original

            # Recalculate distance to new centroid if cluster changed
            if new_cid != old_cid:
                # Find post embedding
                post_id = assignment['post_id']
                post_idx = next((i for i, p in enumerate(posts_data) if p['id'] == post_id), None)
                if post_idx is not None and post_idx in index_map:
                    embedding_idx = index_map.index(post_idx)
                    post_embedding = embeddings[embedding_idx]
                    new_centroid = centroids[new_cid]
                    new_distance = np.linalg.norm(post_embedding - new_centroid)
                else:
                    new_distance = assignment['distance']  # Fallback
            else:
                new_distance = assignment['distance']

            new_assignments.append({
                'post_id': assignment['post_id'],
                'cluster_id': new_cid,
                'distance': float(new_distance)
            })

        # Update centroid examples - remove small clusters and keep large ones
        new_centroid_examples = [
            ex for ex in clustering_result['centroid_examples']
            if ex['cluster_id'] not in small_clusters
        ]

        # Update cluster count
        new_n_clusters = len(large_clusters)

        logging.info(f"Merged {len(small_clusters)} small clusters. Final cluster count: {new_n_clusters}")

        return {
            'assignments': new_assignments,
            'centroid_examples': new_centroid_examples,
            'n_clusters': new_n_clusters,
            'centroids': centroids,
            'merged_clusters': merge_map  # Record what was merged
        }

    def cluster_posts(self, posts_data, num_clusters=None):
        """Run KMeans clustering on Doc2Vec embeddings and surface centroid exemplars."""
        embeddings = []
        index_map = []
        for idx, post in enumerate(posts_data):
            vector = post.get('embedding')
            if vector:
                safe_vector = np.nan_to_num(np.array(vector, dtype=float)).tolist()
                embeddings.append(safe_vector)
                index_map.append(idx)

        if len(embeddings) < 2:
            logging.warning("Skipping clustering: not enough embeddings available")
            return None

        if num_clusters is None:
            # Use Elbow Method to find optimal number of clusters
            num_clusters = self._find_optimal_clusters(embeddings)

        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_

            cluster_assignments = []
            centroid_examples = []

            for cluster_id in range(num_clusters):
                member_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                if not member_indices:
                    continue

                distances = []
                for idx in member_indices:
                    vector = np.array(embeddings[idx])
                    centroid = centroids[cluster_id]
                    distance = np.linalg.norm(vector - centroid)
                    distances.append((idx, distance))

                distances.sort(key=lambda x: x[1])
                closest_idx, closest_distance = distances[0]
                post_idx = index_map[closest_idx]
                post = posts_data[post_idx]

                centroid_examples.append({
                    'cluster_id': cluster_id,
                    'post_id': post['id'],
                    'title': post['title'],
                    'cleaned_content': post['cleaned_content'],
                    'distance': float(closest_distance)
                })

                for idx, distance in distances:
                    post_idx = index_map[idx]
                    cluster_assignments.append({
                        'post_id': posts_data[post_idx]['id'],
                        'cluster_id': cluster_id,
                        'distance': float(distance)
                    })

            # Post-process: merge small outlier clusters
            result = {
                'assignments': cluster_assignments,
                'centroid_examples': centroid_examples,
                'n_clusters': num_clusters,
                'centroids': centroids
            }

            result = self._merge_small_clusters(result, embeddings, index_map, posts_data, min_cluster_size=3)

            return result
        except Exception as cluster_error:
            logging.error(f"Clustering failed: {cluster_error}")
            return None
    
    def generate_processing_summary(self, posts_data, subreddit_name, clustering_result=None, doc2vec_vector_size=None):
        """Generate comprehensive summary for processed data"""
        if not posts_data:
            return "No data processed"
        
        # Basic statistics
        total_posts = len(posts_data)
        total_keywords = sum(len(post['keywords']) for post in posts_data)
        posts_with_images = sum(1 for post in posts_data if post['features']['has_image'])
        
        # Content analysis
        avg_text_length = np.mean([post['features']['text_length'] for post in posts_data])
        avg_score = np.mean([post['features']['score'] for post in posts_data])
        
        # Time range
        timestamps = [post['features']['created_utc'] for post in posts_data]
        time_range = {
            'earliest': datetime.fromtimestamp(min(timestamps)),
            'latest': datetime.fromtimestamp(max(timestamps))
        }
        
        # Most common keywords
        all_keywords = []
        for post in posts_data:
            all_keywords.extend(post['keywords'])
        
        keyword_counter = defaultdict(int)
        for keyword in all_keywords:
            keyword_counter[keyword] += 1
        
        top_keywords = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        
        cluster_section = "No clustering performed"
        if clustering_result and clustering_result.get('centroid_examples'):
            lines = [
                f"  Cluster {item['cluster_id']}: {item['title'][:80]}... (distance {item['distance']:.3f})"
                if item['title'] else
                f"  Cluster {item['cluster_id']}: {item['cleaned_content'][:80]}... (distance {item['distance']:.3f})"
                for item in clustering_result['centroid_examples']
            ]
            cluster_section = "\n".join(lines)

        summary = f"""
=== DATA PROCESSING SUMMARY ===
Session ID: {self.session_id}
Subreddit: r/{subreddit_name}
Total Posts Processed: {total_posts}

Content Statistics:
- Average text length: {avg_text_length:.1f} characters
- Average score: {avg_score:.1f}
- Posts with images: {posts_with_images} ({posts_with_images/total_posts*100:.1f}%)
- Total keywords extracted: {total_keywords}
- Average keywords per post: {total_keywords/total_posts:.1f}

Time Range:
- Earliest post: {time_range['earliest']}
- Latest post: {time_range['latest']}

Top Keywords:
{chr(10).join([f"  {i+1}. {keyword} ({count})" for i, (keyword, count) in enumerate(top_keywords)])}

Data Structure:
- Optimized for clustering algorithms
- Ready for doc2vec embedding
- Comprehensive feature extraction
- Privacy-protected user data
- Doc2Vec vector size: {doc2vec_vector_size if doc2vec_vector_size else 'N/A'}

Clustering:
- Number of clusters: {clustering_result['n_clusters'] if clustering_result else 'N/A'}
- Centroid exemplars:
{cluster_section}
        """

        return summary
    
    def process_data(self, subreddit_name, num_posts, save_format='all', recluster_all=True):
        """Main data processing pipeline

        Args:
            subreddit_name: Name of subreddit to scrape
            num_posts: Number of posts to fetch
            save_format: Format to save data ('json', 'pickle', 'clustering', 'all')
            recluster_all: If True, load existing posts and recluster everything together
        """
        logging.info(f"Starting data processing: r/{subreddit_name}, {num_posts} posts (recluster_all={recluster_all})")

        # Fetch data based on request size
        start_time = time.time()

        if num_posts > 1000:
            posts_data = self.fetch_large_dataset(subreddit_name, num_posts, batch_timeout=60)
        else:
            posts_data = self.fetch_posts_batch(subreddit_name, num_posts, timeout=60)

        processing_time = time.time() - start_time

        if not posts_data:
            logging.error("No data retrieved")
            return None

        # Generate TF-IDF embeddings (optional analytics)
        embedding_feature_names = None
        vectorizer = None
        tfidf_info = self.generate_tfidf_embeddings(posts_data)
        if tfidf_info:
            embedding_feature_names = tfidf_info['feature_names']
            vectorizer = tfidf_info['vectorizer']
            for post, vector in zip(posts_data, tfidf_info['vectors']):
                post['tfidf_vector'] = vector
        else:
            for post in posts_data:
                post['tfidf_vector'] = None

        # Generate Doc2Vec embeddings for clustering requirement
        doc2vec_model = None
        doc2vec_vector_size = None
        doc2vec_info = self.generate_doc2vec_embeddings(posts_data)
        if doc2vec_info:
            doc2vec_model = doc2vec_info['model']
            doc2vec_vector_size = doc2vec_model.vector_size
            vectors_by_id = {doc_id: vector for doc_id, vector in zip(doc2vec_info['ids'], doc2vec_info['vectors'])}
            for post in posts_data:
                vector = vectors_by_id.get(post['id'])
                post['embedding'] = np.nan_to_num(np.array(vector, dtype=float)).tolist() if vector is not None else None
        else:
            for post in posts_data:
                post['embedding'] = None

        # Persist processed posts to database FIRST (before clustering)
        self.save_posts_to_database(posts_data)

        # Run clustering using Doc2Vec embeddings
        if recluster_all:
            # Load existing posts and recluster everything together
            logging.info("Loading existing posts for full reclustering...")
            existing_posts = self.load_existing_posts_from_database()

            # Merge with new posts
            all_posts = self.merge_posts(existing_posts, posts_data)

            # Recluster all posts
            logging.info(f"Reclustering {len(all_posts)} total posts...")
            clustering_result = self.cluster_posts(all_posts)
            cluster_assignments = clustering_result['assignments'] if clustering_result else []

            # Delete old cluster assignments and save new ones
            if cluster_assignments:
                logging.info("Deleting old cluster assignments...")
                self.delete_all_cluster_assignments()
                logging.info(f"Saving {len(cluster_assignments)} new cluster assignments...")
                self.save_clusters_to_database(cluster_assignments, self.session_id)
        else:
            # Just cluster the new batch (old behavior, creates duplicates)
            clustering_result = self.cluster_posts(posts_data)
            cluster_assignments = clustering_result['assignments'] if clustering_result else []
            if cluster_assignments:
                self.save_clusters_to_database(cluster_assignments, self.session_id)

        logging.info(f"Data collection completed in {processing_time:.1f} seconds")
        logging.info(f"Successfully processed {len(posts_data)} posts")

        # Save processed data
        saved_files = self.save_processed_data(
            posts_data,
            subreddit_name,
            save_format,
            embedding_feature_names=embedding_feature_names,
            vectorizer=vectorizer,
            doc2vec_vector_size=doc2vec_vector_size
        )

        # Persist Doc2Vec model if available
        if doc2vec_model is not None:
            doc2vec_path = self.embeddings_dir / f"{subreddit_name}_{self.session_id}_doc2vec.model"
            doc2vec_model.save(str(doc2vec_path))
            saved_files['doc2vec_model'] = doc2vec_path

        # Save clustering exemplars
        if clustering_result:
            clusters_file = self.metadata_dir / f"{subreddit_name}_{self.session_id}_clusters.json"
            with open(clusters_file, 'w', encoding='utf-8') as f:
                json.dump(clustering_result, f, indent=2)
            saved_files['clusters'] = clusters_file

        # Generate and save summary
        summary = self.generate_processing_summary(
            posts_data,
            subreddit_name,
            clustering_result,
            doc2vec_vector_size=doc2vec_vector_size
        )
        print(summary)

        # Save summary to file
        summary_file = self.metadata_dir / f"{subreddit_name}_{self.session_id}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        logging.info("Data processing pipeline completed successfully")

        return {
            'data': posts_data,
            'files': saved_files,
            'summary': summary,
            'session_id': self.session_id,
            'processing_time': processing_time,
            'embedding_feature_names': embedding_feature_names,
            'database': 'SQLite (reddit_data)',
            'clusters': clustering_result,
            'doc2vec_vector_size': doc2vec_vector_size
        }

def main():
    # Reddit API credentials
    CLIENT_ID = "R4r2pV4_CLRBpr_Csx_F_A"
    CLIENT_SECRET = "ZHMtPtLx-PSYMAriIOXG4hD6XgkqlA"
    USER_AGENT = "Beginning-Split862"
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Reddit Data Processing for Clustering & Automation')
    parser.add_argument('--subreddit', default='iphone', help='Subreddit to scrape')
    parser.add_argument('--posts', type=int, default=5000, help='Number of posts to fetch')
    parser.add_argument('--data-dir', default='reddit_data', help='Data storage directory')
    parser.add_argument('--format', choices=['json', 'pickle', 'clustering', 'all'], 
                       default='all', help='Output format')
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = RedditDataProcessor(CLIENT_ID, CLIENT_SECRET, USER_AGENT, args.data_dir)
    result = processor.process_data(args.subreddit, args.posts, args.format)
    
    if result:
        print(f"\n=== FILES CREATED ===")
        for format_type, filepath in result['files'].items():
            print(f"{format_type.upper()}: {filepath}")
        
        print(f"\nSession ID: {result['session_id']}")
        print(f"Processing time: {result['processing_time']:.1f} seconds")

if __name__ == "__main__":
    main()
