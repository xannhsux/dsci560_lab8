#!/usr/bin/env python3
"""
Enhanced Cluster Visualization for Lab 5
Meets requirements:
- Display K clusters (K-means)
- Show keywords for each cluster
- Verify similarity by displaying and comparing message contents
- Graphical visualization
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from database_connection import SQLiteConnection

HTML_OUTPUT = Path("enhanced_cluster_report.html")

def fetch_cluster_data():
    """Fetch comprehensive cluster data including keywords and content"""
    try:
        db_conn = SQLiteConnection()
    except Exception as e:
        print(f"[ERROR] Could not connect to MySQL database: {e}")
        return None

    # Get total posts
    total_posts_result = db_conn.execute_query("SELECT COUNT(*) FROM posts;", fetch='one')
    total_posts = total_posts_result[0]

    # Get cluster distribution
    dist_results = db_conn.execute_query("""
        SELECT cluster_id, COUNT(*)
        FROM clusters
        GROUP BY cluster_id
        ORDER BY cluster_id;
    """, fetch='all')
    distribution = {cid: count for cid, count in dist_results}

    # Get posts with keywords and content for each cluster
    post_results = db_conn.execute_query("""
        SELECT
            c.cluster_id,
            p.id,
            p.title,
            p.cleaned_content,
            p.keywords,
            p.topics,
            c.distance
        FROM clusters c
        JOIN posts p ON c.post_id = p.id
        ORDER BY c.cluster_id, c.distance ASC
    """, fetch='all')

    clusters = {}
    for cluster_id, post_id, title, content, keywords_json, topics_json, distance in post_results:
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                'posts': [],
                'all_keywords': [],
                'all_topics': []
            }

        # Parse JSON keywords and topics
        try:
            keywords = json.loads(keywords_json) if keywords_json else []
            topics = json.loads(topics_json) if topics_json else []
        except:
            keywords = []
            topics = []

        clusters[cluster_id]['posts'].append({
            'id': post_id,
            'title': title or '',
            'content': content or '',
            'keywords': keywords,
            'topics': topics,
            'distance': distance
        })
        clusters[cluster_id]['all_keywords'].extend(keywords)
        clusters[cluster_id]['all_topics'].extend(topics)

    db_conn.close()

    # Calculate top keywords for each cluster
    for cluster_id, data in clusters.items():
        keyword_counts = Counter(data['all_keywords'])
        data['top_keywords'] = keyword_counts.most_common(10)

        topic_counts = Counter(data['all_topics'])
        data['top_topics'] = topic_counts.most_common(10)

    return {
        'total_posts': total_posts,
        'distribution': distribution,
        'clusters': clusters
    }

def create_distribution_chart(distribution, output_path='cluster_distribution.png'):
    """Create bar chart of cluster distribution"""
    cluster_ids = sorted(distribution.keys())
    counts = [distribution[cid] for cid in cluster_ids]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_ids, counts, color='steelblue', alpha=0.8, edgecolor='navy')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Cluster ID', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Posts', fontsize=12, fontweight='bold')
    plt.title('K-Means Clustering Distribution (K=10)', fontsize=14, fontweight='bold')
    plt.xticks(cluster_ids)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved distribution chart: {output_path}")
    return output_path

def generate_html_report(data):
    """Generate comprehensive HTML report with keywords and content comparison"""

    # Create distribution chart
    chart_path = create_distribution_chart(data['distribution'])

    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 24px;
            color: #111;
            background: #fafafa;
        }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 32px; border-radius: 12px; }
        h1 { color: #1a1a1a; margin-bottom: 8px; }
        h2 { color: #2563eb; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }
        h3 { color: #3730a3; margin-top: 24px; }
        .muted { color: #666; font-size: 14px; }
        .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 { color: white; margin: 0; font-size: 14px; text-transform: uppercase; }
        .stat-card .value { font-size: 36px; font-weight: bold; margin: 8px 0; }

        .cluster-card {
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 24px;
            margin: 24px 0;
            background: #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .cluster-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .cluster-id {
            font-size: 24px;
            font-weight: bold;
            color: #3730a3;
        }
        .post-count {
            background: #eef2ff;
            color: #3730a3;
            padding: 6px 16px;
            border-radius: 999px;
            font-weight: 600;
        }

        .keywords-section {
            background: #f0fdf4;
            border-left: 4px solid #16a34a;
            padding: 16px;
            margin: 16px 0;
            border-radius: 8px;
        }
        .keyword-tag {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            padding: 4px 12px;
            margin: 4px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 500;
        }
        .keyword-tag .count {
            background: #16a34a;
            color: white;
            padding: 2px 6px;
            border-radius: 999px;
            margin-left: 6px;
            font-size: 11px;
        }

        .posts-section {
            margin-top: 20px;
        }
        .post-item {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
            transition: all 0.2s;
        }
        .post-item:hover {
            border-color: #3730a3;
            box-shadow: 0 4px 12px rgba(55, 48, 163, 0.1);
        }
        .post-title {
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 8px;
            font-size: 15px;
        }
        .post-content {
            color: #4b5563;
            font-size: 14px;
            line-height: 1.6;
            margin: 8px 0;
            max-height: 120px;
            overflow: hidden;
        }
        .post-meta {
            font-size: 12px;
            color: #9ca3af;
            margin-top: 8px;
        }
        .distance-badge {
            background: #fef3c7;
            color: #92400e;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }

        .chart-container {
            margin: 32px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .similarity-note {
            background: #eff6ff;
            border-left: 4px solid #2563eb;
            padding: 16px;
            margin: 16px 0;
            border-radius: 8px;
        }
        .similarity-note strong { color: #1e40af; }
    </style>
    """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Enhanced Reddit Clustering Report</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>🎯 Enhanced K-Means Clustering Report</h1>
        <p class="muted">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="overview-grid">
            <div class="stat-card">
                <h3>Total Posts</h3>
                <div class="value">{data['total_posts']:,}</div>
            </div>
            <div class="stat-card">
                <h3>Number of Clusters</h3>
                <div class="value">{len(data['clusters'])}</div>
            </div>
            <div class="stat-card">
                <h3>Algorithm</h3>
                <div class="value" style="font-size: 24px;">K-Means</div>
            </div>
        </div>

        <h2>[DATA] Cluster Distribution Visualization</h2>
        <div class="chart-container">
            <img src="{chart_path}" alt="Cluster Distribution">
        </div>
"""

    # Generate cluster sections
    for cluster_id in sorted(data['clusters'].keys()):
        cluster_data = data['clusters'][cluster_id]
        post_count = len(cluster_data['posts'])

        html += f"""
        <div class="cluster-card">
            <div class="cluster-header">
                <div class="cluster-id">Cluster {cluster_id}</div>
                <div class="post-count">{post_count} posts</div>
            </div>

            <div class="keywords-section">
                <h3>🔑 Top Keywords (Common Themes)</h3>
                <div>
"""

        # Add keywords
        for keyword, count in cluster_data['top_keywords'][:15]:
            html += f'<span class="keyword-tag">{keyword}<span class="count">{count}</span></span>'

        html += """
                </div>
            </div>

            <div class="similarity-note">
                <strong>📝 Similarity Verification:</strong> The posts below share common keywords and were grouped by Doc2Vec semantic similarity.
                Posts closest to the cluster centroid (lowest distance) best represent this cluster's themes.
            </div>

            <div class="posts-section">
                <h3>📄 Representative Posts (Closest to Centroid)</h3>
"""

        # Add top 5 posts closest to centroid
        for i, post in enumerate(cluster_data['posts'][:5], 1):
            content_preview = post['content'][:300] + '...' if len(post['content']) > 300 else post['content']

            html += f"""
                <div class="post-item">
                    <div class="post-title">#{i}: {post['title']}</div>
                    <div class="post-content">{content_preview}</div>
                    <div class="post-meta">
                        Distance from centroid: <span class="distance-badge">{post['distance']:.3f}</span>
                        | Keywords: {', '.join(post['keywords'][:5])}
                    </div>
                </div>
"""

        html += """
            </div>
        </div>
"""

    html += """
        <div style="margin-top: 48px; padding: 24px; background: #f9fafb; border-radius: 12px; text-align: center;">
            <p style="color: #6b7280; font-size: 14px;">
                 Generated with K-Means clustering on Doc2Vec embeddings<br>
                Lab 5: Reddit Data Processing & Clustering System
            </p>
        </div>
    </div>
</body>
</html>
"""

    return html

def main():
    print("=" * 60)
    print("Enhanced Cluster Visualization for Lab 5")
    print("=" * 60)

    # Fetch data
    print("\n[DATA] Fetching cluster data from database...")
    data = fetch_cluster_data()

    if not data:
        return

    print(f"[SUCCESS] Loaded {data['total_posts']} posts across {len(data['clusters'])} clusters")

    # Generate HTML report
    print("\nGenerating enhanced visualization...")
    html_content = generate_html_report(data)

    # Save HTML
    HTML_OUTPUT.write_text(html_content, encoding='utf-8')
    print(f"[SUCCESS] Saved enhanced report: {HTML_OUTPUT.resolve()}")

    # Print cluster summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    for cluster_id in sorted(data['clusters'].keys()):
        cluster_data = data['clusters'][cluster_id]
        keywords = [kw for kw, _ in cluster_data['top_keywords'][:5]]
        print(f"Cluster {cluster_id}: {len(cluster_data['posts'])} posts")
        print(f"  Top keywords: {', '.join(keywords)}")

    print("\nDone! Open the HTML file to view the visualization.")

if __name__ == "__main__":
    main()