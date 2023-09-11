import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from extraction.init_center import kmeans_init


def kmeans_silhouette(features):
    # calculate sqrt(n)
    sqrt_n = int(np.sqrt(len(features)))
    print(sqrt_n)
    # Initialise k and clustering results
    k = sqrt_n
    best_k = k
    best_clusters = None
    best_avg_silhouette = -1

    # Results of the selection of the initial center
    clusters, centers = kmeans_init(features)
    best_centers = None  # 初始化best_centers
    center_indices = None  # 初始化center_indices
    print(centers.shape)
    # Iterative Procedure
    while k > 2:
        # Calculate the Euclidean distance between cluster centers
        cluster_centers = centers
        distances = cdist(cluster_centers, cluster_centers, metric='euclidean')

        # Find the indexes of the two the nearest clusters
        min_distance = np.inf
        merge_cluster_indices = None
        # Iterate over the values in the upper right corner of the matrix
        for i in range(k):
            for j in range(i + 1, k):
                if distances[i, j] < min_distance:
                    min_distance = distances[i, j]
                    merge_cluster_indices = (i, j)

        # Merge the two the nearest clusters and change the high cluster number to the low cluster number
        merged_cluster = np.where(clusters == merge_cluster_indices[1], merge_cluster_indices[0], clusters)

        # Update clustering results
        clusters = np.where(merged_cluster > merge_cluster_indices[1], merged_cluster - 1, merged_cluster)

        # Update the cluster center, selecting the actual data point as the new cluster center
        new_centers = []
        for cluster_id in range(k - 1):
            # Get samples of the current cluster
            cluster_samples = features[clusters == cluster_id]
            # Calculate the current cluster mean
            cluster_mean = np.mean(cluster_samples, axis=0)
            # Calculate the Euclidean distance between the sample and the centre point to find the actual center
            distances = np.linalg.norm(cluster_samples - cluster_mean, axis=1)
            closest_sample_index = np.argmin(distances)
            # Choose the nearest sample as the new cluster centroid
            new_centers.append(cluster_samples[closest_sample_index])

        centers = new_centers
        # update number of cluster
        k -= 1
        # print(len(centers))
        # Calculate Silhouette Coefficient
        avg_silhouette = silhouette_score(features, clusters)


        # center_indices = []
        # 更新最佳结果
        if avg_silhouette > best_avg_silhouette:
            best_avg_silhouette = avg_silhouette
            best_k = k
            best_clusters = clusters.copy()
            best_centers = centers.copy()
            center_indices = []
            for cluster_center in best_centers:
                center_index = np.where((features == cluster_center).all(axis=1))[0][0]
                center_indices.append(center_index)

    # return result
    print("best_k:" + str(best_k))
    print("best_clusters:" + str(best_clusters))
    if best_centers is not None:
        print("best_centers:(Length: " + str(len(best_centers)) + ")")
    else:
        print("best_centers is None, cannot determine its length.")
    print("best_avg_sc:" + str(best_avg_silhouette))
    print("best_center_index:" + str(center_indices))

    return best_clusters, best_centers, best_k, center_indices

