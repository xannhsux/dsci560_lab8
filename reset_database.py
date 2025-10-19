#!/usr/bin/env python3
"""Reset database - delete all data and recreate fresh"""

from database_connection import SQLiteConnection
import os
from pathlib import Path

print("=" * 70)
print("DATABASE RESET")
print("=" * 70)

# Delete SQLite file if it exists
sqlite_file = Path("reddit_data/reddit_posts.db")
if sqlite_file.exists():
    sqlite_file.unlink()
    print(f"\n[DELETED] Removed SQLite database: {sqlite_file}")
else:
    print(f"\n[INFO] No SQLite database found")

# Connect to SQLite and drop all data
db = SQLiteConnection()

# Drop tables
print("\n[DROPPING] Dropping existing tables...")
db.execute_query("DROP TABLE IF EXISTS clusters", fetch=None)
db.execute_query("DROP TABLE IF EXISTS posts", fetch=None)
print("[SUCCESS] Dropped all tables")

# Recreate tables
print("\n[CREATING] Creating fresh tables...")

# Create posts table
db.execute_query("""
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
    )
""", fetch=None)

# Create indexes for posts table
db.execute_query("CREATE INDEX IF NOT EXISTS idx_session ON posts(session_id)", fetch=None)
db.execute_query("CREATE INDEX IF NOT EXISTS idx_subreddit ON posts(subreddit)", fetch=None)
db.execute_query("CREATE INDEX IF NOT EXISTS idx_created ON posts(created_datetime)", fetch=None)

# Create clusters table
db.execute_query("""
    CREATE TABLE IF NOT EXISTS clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT,
        session_id TEXT,
        cluster_id INTEGER,
        distance REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
    )
""", fetch=None)

# Create indexes for clusters table
db.execute_query("CREATE INDEX IF NOT EXISTS idx_cluster ON clusters(cluster_id)", fetch=None)
db.execute_query("CREATE INDEX IF NOT EXISTS idx_post ON clusters(post_id)", fetch=None)
db.execute_query("CREATE INDEX IF NOT EXISTS idx_session ON clusters(session_id)", fetch=None)

print("[SUCCESS] Created fresh tables")

# Verify empty
result = db.execute_query("SELECT COUNT(*) FROM posts", fetch='one')
posts_count = result[0]
result = db.execute_query("SELECT COUNT(*) FROM clusters", fetch='one')
clusters_count = result[0]

print(f"\n[VERIFY] Posts table: {posts_count} rows")
print(f"[VERIFY] Clusters table: {clusters_count} rows")

db.close()

print("\n" + "=" * 70)
print("[SUCCESS] Database reset complete - ready for fresh data")
print("=" * 70)
print("\nNext step: Run 'python direct_iphone_processing.py' to collect 5000 posts")