import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load dataset from CSV file"""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Select relevant features and scale them"""
    X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

def apply_kmeans(X_scaled, n_clusters=5):
    """Apply K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans

def visualize_clusters(X, clusters):
    """Visualize clustered data"""
    plt.figure()
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segmentation using K-Means")
    plt.savefig("clusters.png")  # saves image
    plt.show()

def main():
    data = load_data("Mall_Customer_Segmentation_Data.csv")
    X_scaled, X = preprocess_data(data)
    clusters, model = apply_kmeans(X_scaled)
    data['Cluster'] = clusters
    visualize_clusters(X, clusters)

if __name__ == "__main__":
    main()