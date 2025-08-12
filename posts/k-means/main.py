# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import seaborn as sns
import numpy as np

# Create synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])

# Calculate inertia for different values of K
inertia = []
k_range = range(2, 11)  # Testing K from 2 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(k_range)
plt.grid()
plt.show()

# %%

class SimpleKMeans:
    def __init__(self, n_clusters=4, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.permutation(X.shape[0])
        self.centroids = X[random_indices[:self.n_clusters]]

    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
        return new_centroids

    def fit(self, X):
        self._initialize_centroids(X)
        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, self.labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self

# Using the same synthetic data from before
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Fit the custom KMeans model
custom_kmeans = SimpleKMeans(n_clusters=4, random_state=42)
custom_kmeans.fit(X)
custom_labels = custom_kmeans.labels

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=custom_labels, palette='viridis', s=100)
plt.scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering from Scratch')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()

# %%
df = pd.read_csv("percentile_rankings.csv")
df.sort_values("xwoba", ascending=False).head(5)

# %%
features = ["xba", "xslg", "xiso", "xobp"]
df_k = df[features]
df_k = df_k.dropna()
df_k = df_k.sort_values("xslg", ascending=False)
df_k.head(5)

# %%

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_k)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

# %%
optimal_k = 6
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init='auto', random_state=42)
clusters = final_kmeans.fit_predict(df_k)

cluster_names = {2: "Elite Slugger", 5: "High-Average Hitter", 3: "Contact Specialist", 4: "Three True Outcome Hitter", 0: "Low-Average Power Threat", 1: "Struggling Hitter"}

# Add the cluster labels back to a copy of our original dataframe
results_df = df_k.loc[df_k.index].copy()
results_df['cluster'] = clusters

# map cluster names to the cluster labels
results_df['cluster'] = results_df['cluster'].map(cluster_names)

cluster_profiles = results_df.groupby('cluster')[features].mean().round(1)
cluster_profiles = cluster_profiles.sort_values(by=features, ascending=False)

print("--- Cluster Profiles (Average Percentiles) ---")
cluster_profiles.to_clipboard()

cluster_profiles = cluster_profiles.rename(index=cluster_names)
cluster_profiles

# %%
# Data from the user
data = {
    'cluster': ['Elite Slugger', 'High-Average Hitter', 'Contact Specialist', 'Three True Outcome Hitter', 'Low-Average Power Threat', 'Struggling Hitter'],
    'xba': [80.5, 75.8, 59.0, 56.2, 30.3, 17.2],
    'xslg': [91.8, 64.9, 26.0, 81.8, 50.8, 14.3],
    'xiso': [89.3, 55.8, 18.3, 82.8, 59.0, 20.5],
    'xobp': [89.7, 72.8, 60.4, 40.5, 32.6, 18.6]
}
df_vis = pd.DataFrame(data)
df_vis = df_vis.set_index('cluster')

# Number of variables we're plotting.
num_vars = len(df_vis.columns)

# Compute angle for each axis.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop"
# and append the start to the end.
angles += angles[:1]

# Labels for each axis
labels = df_vis.columns

# Create the figure and subplots
fig, axes = plt.subplots(figsize=(10, 9), nrows=3, ncols=2, subplot_kw=dict(polar=True))
axes = axes.flatten() # Flatten the 3x2 grid of axes for easy iteration

# Define colors for each cluster
colors = plt.cm.viridis(np.linspace(0, 1, len(df_vis)))

# Plot each cluster on a separate subplot
for i, (cluster_name, row) in enumerate(df_vis.iterrows()):
    ax = axes[i]
    values = row.tolist()
    values += values[:1]  # complete the loop

    # Plot the data
    ax.plot(angles, values, color=colors[i], linewidth=2)
    ax.fill(angles, values, color=colors[i], alpha=0.25)

    # Prettify the plot
    ax.set_rlim(0, 100) # Set radial limits to be consistent (0-100 for percentiles)
    ax.set_yticklabels([0, 25, 50, 75, 100])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(cluster_name, size=12, y=1.1)

# Adjust layout to prevent titles from overlapping
plt.tight_layout(pad=3.0)
plt.show()

# %%
results_df = results_df.sort_values('xslg', ascending=False)
results_df = results_df.merge(df[['player_name']], left_index=True, right_index=True, how='left')
results_df[results_df['cluster'] == "Elite Slugger"].sort_values('xslg', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "High-Average Hitter"].sort_values('xba', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "Contact Specialist"].sort_values('xba', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "Three True Outcome Hitter"].sort_values('xslg', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "Low-Average Power Threat"].sort_values('xba', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "Struggling Hitter"].sort_values('xba', ascending=False).head(5)

# %%
results_df[results_df['cluster'] == "Struggling Hitter"].sort_values('xba', ascending=False).tail(5)

# %%
results_df[results_df['cluster'] == "Struggling Hitter"].sort_values('xba', ascending=False).tail(5)

# %%
results_df[results_df['cluster'] == "Struggling Hitter"].sort_values('xba', ascending=False).tail(5)

# %%
results_df[results_df['cluster'] == "Struggling Hitter"].sort_values('xba', ascending=False).tail(5)


