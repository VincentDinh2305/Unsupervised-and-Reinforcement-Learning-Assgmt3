# Dinh Hoang Viet Phuong - 301123263


# import all necessary libraries
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.metrics import silhouette_score


# 1. Retrieve and load the Olivetti faces dataset

# Fetch the Olivetti faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
faces = data.images
X = data.data
y = data.target

# Display the first few images as an example
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(faces[i], cmap='gray')
    plt.title(f"Face {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# 2. Split the training set, a validation set, and a test set using stratified sampling to ensure 
# that there are the same number of images per person in each set.

# Split the dataset into a temporary training set and a test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Split the temporary training set into a final training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("Training set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))


# 3. Using k-fold cross validation, train a classifier to predict which person is represented in 
# each picture, and evaluate it on the validation set

# Choose a classifier
clf = SVC(kernel='linear', random_state=42)

# 5-fold cross-validation on the training data
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation scores: \n", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())

# Train the classifier on the entire training set
clf.fit(X_train, y_train)

# Evaluate the classifier on the validation set
val_score = clf.score(X_val, y_val)
print("Validation set accuracy:", val_score)


# 4. Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering 
# (DHC) and using the centroid-based clustering rule, reduce the dimensionality of the set by using 
# the following similarity measures:
    
# Compute cosine distance matrix
def cosine_distance_matrix(X):
    return np.array([[cosine(x,y) for y in X] for x in X])

# Number of clusters
n_clusters = 40

# a) Euclidean Distance
ahc_euclidean = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
clusters_euclidean = ahc_euclidean.fit_predict(X)

# b) Manhattan Distance
ahc_manhattan = AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='average')
clusters_manhattan = ahc_manhattan.fit_predict(X)

# c) Cosine Similarity
cosine_distances = cosine_distance_matrix(X)
ahc_cosine = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
clusters_cosine = ahc_cosine.fit_predict(cosine_distances)

# Print cluster assignments
print("Sample clusters with Euclidean distance: \n", clusters_euclidean[:20])
print("Sample clusters with Manhattan distance: \n", clusters_manhattan[:20])
print("Sample clusters with Cosine similarity: \n", clusters_cosine[:20])


# 5. Use the silhouette score approach to choose the number of clusters for 4(a), 4(b), and 4(c)

# Function to compute silhouette scores for a range of cluster numbers
def compute_silhouette_scores(X, distance_metric):
    cluster_range = range(2, 41)
    silhouette_scores = []

    for n_clusters in cluster_range:
        if distance_metric == "euclidean":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        elif distance_metric == "manhattan":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='manhattan', linkage='average')
        elif distance_metric == "cosine":
            cosine_distances = cosine_distance_matrix(X)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
            X = cosine_distances

        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    return cluster_range, silhouette_scores

# Compute silhouette scores
cluster_range, silhouette_scores_euclidean = compute_silhouette_scores(X, "euclidean")
_, silhouette_scores_manhattan = compute_silhouette_scores(X, "manhattan")
_, silhouette_scores_cosine = compute_silhouette_scores(X, "cosine")

# Plot
plt.figure(figsize=(15, 7))
plt.plot(cluster_range, silhouette_scores_euclidean, label='Euclidean', color='blue')
plt.plot(cluster_range, silhouette_scores_manhattan, label='Manhattan', color='green')
plt.plot(cluster_range, silhouette_scores_cosine, label='Cosine', color='red')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()
  

# 6. Use the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation
    
# Using the cluster assignments from 4(a) (Euclidean distance) as features
X_new = ahc_euclidean.labels_.reshape(-1, 1)

# Choose a classifier. Here, I'll use the Support Vector Machine (SVM) classifier with a linear kernel.
clf = SVC(kernel='linear', random_state=42)

# 5-fold cross-validation on the new feature set
scores = cross_val_score(clf, X_new, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())    
    
    
    
