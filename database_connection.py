#!/usr/bin/env python3
"""
SQLite Database Connection Handler for Reddit Data Processing
Provides SQLite connectivity and connection management for the Reddit data processing system.
"""

import sqlite3
import logging
import os
from pathlib import Path
from contextlib import contextmanager

class SQLiteConnection:
    """Handle SQLite database connections for Reddit data processing"""

    def __init__(self, db_path=None):
        """Initialize SQLite connection parameters"""
        if db_path is None:
            # Default to reddit_data/reddit_posts.db as per README
            db_dir = Path("reddit_data")
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "reddit_posts.db"

        self.db_path = str(db_path)
        self.connection = None
        self._connect()
        self._initialize_schema()

    def _connect(self):
        """Establish connection to SQLite database"""
        try:
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.connection.row_factory = sqlite3.Row
            # Enable foreign keys
            self.connection.execute("PRAGMA foreign_keys = ON")
            # Improve performance
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")

            logging.info(f"Successfully connected to SQLite database: {self.db_path}")
            return True

        except sqlite3.Error as e:
            logging.error(f"Error connecting to SQLite: {e}")
            self.connection = None
            return False

    def _initialize_schema(self):
        """Create tables if they don't exist"""
        if not self.connection:
            return

        try:
            cursor = self.connection.cursor()

            # Create posts table
            cursor.execute("""
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
            """)

            # Create clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    post_id TEXT,
                    session_id TEXT,
                    cluster_id INTEGER,
                    distance REAL,
                    PRIMARY KEY(post_id, session_id),
                    FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_session
                ON posts(session_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id
                ON clusters(cluster_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_clusters_session
                ON clusters(session_id)
            """)

            self.connection.commit()
            logging.info("Database schema initialized successfully")

        except sqlite3.Error as e:
            logging.error(f"Error initializing schema: {e}")
            if self.connection:
                self.connection.rollback()

    def execute_query(self, query, params=None, fetch='none'):
        """
        Execute a SQL query with optional parameters

        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
            fetch: 'one', 'all', or 'none'

        Returns:
            Query results based on fetch parameter
        """
        if not self.connection:
            logging.error("No database connection available")
            return None

        try:
            cursor = self.connection.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch == 'one':
                result = cursor.fetchone()
                return result
            elif fetch == 'all':
                result = cursor.fetchall()
                return result
            else:
                # For INSERT/UPDATE/DELETE, commit and return affected rows
                self.connection.commit()
                return cursor.rowcount

        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            logging.error(f"Query: {query}")
            if self.connection:
                self.connection.rollback()
            return None

    def execute_many(self, query, params_list):
        """
        Execute a query multiple times with different parameters

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of affected rows
        """
        if not self.connection:
            logging.error("No database connection available")
            return 0

        try:
            cursor = self.connection.cursor()
            cursor.executemany(query, params_list)
            self.connection.commit()
            return cursor.rowcount

        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            if self.connection:
                self.connection.rollback()
            return 0

    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        if not self.connection:
            raise sqlite3.Error("No database connection available")

        try:
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logging.error(f"Transaction failed: {e}")
            raise

    def close(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logging.info("Database connection closed")
            except sqlite3.Error as e:
                logging.error(f"Error closing connection: {e}")

    def __del__(self):
        """Cleanup on object destruction"""
        self.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()