import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

# 设定参数
k_neighbors = 10  # 最近邻个数
feature_save_path = "./features"

# 加载训练集 BoF 直方图
def load_bof_features(filename):
    with open(os.path.join(feature_save_path, filename), "rb") as f:
        return pickle.load(f)

sift_train_bof, sift_train_labels = load_bof_features("sift_train_fv.pkl")
sift_test_bof, sift_test_labels = load_bof_features("sift_test_fv.pkl")

# KNN 进行检索
print("进行 KNN 车辆检索...")
nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(sift_train_bof)

retrieval_results = {}
start_time = time.time()
for i, test_vector in enumerate(sift_test_bof):
    distances, indices = nbrs.kneighbors([test_vector])
    retrieved_labels = sift_train_labels[indices[0]]
    retrieval_results[i] = (retrieved_labels, distances[0])
retrieval_time = time.time() - start_time
print(f"KNN 检索完成，耗时 {retrieval_time:.2f} 秒")

# 评估检索性能
def compute_metrics(retrieval_results, test_labels):
    total_relevant = 0
    total_retrieved = 0
    avg_precision = 0

    for i, (retrieved_labels, distances) in retrieval_results.items():
        relevant_count = np.sum(retrieved_labels == test_labels[i])
        total_relevant += relevant_count
        total_retrieved += len(retrieved_labels)

        # 计算 Average Precision (AP)
        precision_at_k = [np.sum(retrieved_labels[:k] == test_labels[i]) / (k + 1) for k in range(len(retrieved_labels))]
        ap = np.mean(precision_at_k)
        avg_precision += ap

    recall = total_relevant / len(retrieval_results) / k_neighbors
    precision = total_relevant / total_retrieved
    mean_ap = avg_precision / len(retrieval_results)

    return recall, precision, mean_ap

recall, precision, mean_ap = compute_metrics(retrieval_results, sift_test_labels)
print(f"召回率: {recall:.4f}")
print(f"精度: {precision:.4f}")
print(f"MAP: {mean_ap:.4f}")
print(f"检索时间: {retrieval_time:.2f} 秒")
