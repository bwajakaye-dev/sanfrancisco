import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.utils.config import DATA_PROCESSED, RESULTS_DIR, FIGURES_DIR


def perform_clustering_analysis(input_file=None, output_dir=RESULTS_DIR):
    """
    Perform K-Means and DBSCAN clustering on EV adoption data.
    
    Args:
        input_file: Path to features file
        output_dir: Directory to save results
    
    Returns:
        tuple: (clustered DataFrame, kmeans model, dbscan model)
    """
    print("="*60)
    print("Clustering Analysis")
    print("="*60)
    
    # Load data
    if input_file is None:
        input_file = os.path.join(DATA_PROCESSED, 'sf_ev_features.csv')
    
    df = pd.read_csv(input_file)
    print(f"\n📊 Loaded data: {df.shape}")
    
    # Filter for most recent year
    most_recent_year = df['year'].max()
    df_recent = df[df['year'] == most_recent_year].copy()
    print(f"   Using {most_recent_year} data: {len(df_recent)} ZIP codes")
    
    # Select features for clustering
    features = ['ev_percentage', 'total_vehicles', 'ev_growth']
    X = df_recent[features].fillna(0)
    
    print(f"\n📈 Feature statistics:")
    print(X.describe())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means Clustering
    print("\n🔍 K-Means Clustering...")
    
    # Find optimal k using elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_scaled)
        inertias.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
    
    # Use k=4 as default (can be adjusted based on elbow plot)
    optimal_k = 4
    print(f"   Using k={optimal_k} clusters")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_recent['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    # DBSCAN Clustering
    print("\n🔍 DBSCAN Clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    df_recent['cluster_dbscan'] = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(set(df_recent['cluster_dbscan'])) - (1 if -1 in df_recent['cluster_dbscan'] else 0)
    n_noise = list(df_recent['cluster_dbscan']).count(-1)
    print(f"   Found {n_clusters_dbscan} clusters, {n_noise} noise points")
    
    # Analyze K-Means clusters
    print("\n" + "="*60)
    print("K-Means Cluster Analysis")
    print("="*60)
    
    for cluster in sorted(df_recent['cluster_kmeans'].unique()):
        cluster_data = df_recent[df_recent['cluster_kmeans'] == cluster]
        print(f"\n📍 Cluster {cluster}: {len(cluster_data)} ZIP codes")
        print(f"   Avg EV %: {cluster_data['ev_percentage'].mean():.2f}%")
        print(f"   Avg Total Vehicles: {cluster_data['total_vehicles'].mean():.0f}")
        print(f"   Avg EV Growth: {cluster_data['ev_growth'].mean():.2f}%")
        print(f"   ZIP codes: {cluster_data['zip_code'].tolist()}")
    
    # Save results
    output_file = os.path.join(output_dir, 'clustering_results.csv')
    df_recent.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    # Create visualization
    create_cluster_visualization(df_recent, inertias, silhouette_scores, k_range)
    
    return df_recent, kmeans, dbscan


def create_cluster_visualization(df, inertias, silhouette_scores, k_range):
    """Create clustering visualizations."""
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Elbow plot
    axes[0, 0].plot(k_range, inertias, 'bo-')
    axes[0, 0].set_xlabel('Number of Clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Elbow Method')
    axes[0, 0].grid(True)
    
    # Silhouette scores
    axes[0, 1].plot(k_range, silhouette_scores, 'go-')
    axes[0, 1].set_xlabel('Number of Clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Analysis')
    axes[0, 1].grid(True)
    
    # Scatter: EV% vs Total Vehicles
    scatter = axes[1, 0].scatter(
        df['total_vehicles'], 
        df['ev_percentage'],
        c=df['cluster_kmeans'],
        cmap='viridis',
        s=100,
        alpha=0.6
    )
    axes[1, 0].set_xlabel('Total Vehicles')
    axes[1, 0].set_ylabel('EV Percentage (%)')
    axes[1, 0].set_title('K-Means Clusters')
    plt.colorbar(scatter, ax=axes[1, 0], label='Cluster')
    
    # Cluster distribution
    cluster_counts = df['cluster_kmeans'].value_counts().sort_index()
    axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of ZIP Codes')
    axes[1, 1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    output_file = os.path.join(FIGURES_DIR, 'clustering_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    perform_clustering_analysis()