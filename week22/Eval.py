# import os
# import pickle
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# import time

# # 设定参数
# k_neighbors = 10  # 最近邻个数
# feature_save_path = "./features"

# # 加载训练集 BoF 直方图
# def load_bof_features(filename):
#     with open(os.path.join(feature_save_path, filename), "rb") as f:
#         return pickle.load(f)

# sift_train_bof, sift_train_labels = load_bof_features("sift_train_vlad.pkl")
# sift_test_bof, sift_test_labels = load_bof_features("sift_test_vlad.pkl")

# # KNN 进行检索
# print("进行 KNN 车辆检索...")
# nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(sift_train_bof)

# retrieval_results = {}
# start_time = time.time()
# for i, test_vector in enumerate(sift_test_bof):
#     distances, indices = nbrs.kneighbors([test_vector])
#     retrieved_labels = sift_train_labels[indices[0]]
#     retrieval_results[i] = (retrieved_labels, distances[0])
# retrieval_time = time.time() - start_time
# print(f"KNN 检索完成，耗时 {retrieval_time:.2f} 秒")

# # 评估检索性能
# def compute_metrics(retrieval_results, test_labels):
#     total_relevant = 0
#     total_retrieved = 0
#     avg_precision = 0

#     for i, (retrieved_labels, distances) in retrieval_results.items():
#         relevant_count = np.sum(retrieved_labels == test_labels[i])
#         total_relevant += relevant_count
#         total_retrieved += len(retrieved_labels)

#         # 计算 Average Precision (AP)
#         precision_at_k = [np.sum(retrieved_labels[:k] == test_labels[i]) / (k + 1) for k in range(len(retrieved_labels))]
#         ap = np.mean(precision_at_k)
#         avg_precision += ap

#     recall = total_relevant / len(retrieval_results) / k_neighbors
#     precision = total_relevant / total_retrieved
#     mean_ap = avg_precision / len(retrieval_results)

#     return recall, precision, mean_ap

# recall, precision, mean_ap = compute_metrics(retrieval_results, sift_test_labels)
# print(f"召回率: {recall:.4f}")
# print(f"精度: {precision:.4f}")
# print(f"MAP: {mean_ap:.4f}")
# print(f"检索时间: {retrieval_time:.2f} 秒")
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

# 设定参数
k_neighbors = 10  # 最近邻个数
feature_save_path = "./features"

# 加载 BoF 直方图
def load_bof_features(filename):
    with open(os.path.join(feature_save_path, filename), "rb") as f:
        return pickle.load(f)

# 进行 KNN 检索
def knn_retrieval(train_bof, train_labels, test_bof, test_labels):
    print(f"进行 KNN 车辆检索...")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean").fit(train_bof)

    retrieval_results = {}
    start_time = time.time()
    for i, test_vector in enumerate(test_bof):
        distances, indices = nbrs.kneighbors([test_vector])
        retrieved_labels = train_labels[indices[0]]
        retrieval_results[i] = (retrieved_labels, distances[0])
    retrieval_time = time.time() - start_time
    print(f"KNN 检索完成，耗时 {retrieval_time:.2f} 秒")
    return retrieval_results, retrieval_time

# 计算性能指标
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

# 处理所有四种组合
combinations = {
    "SIFT+VLAD": ("sift_train_vlad.pkl", "sift_test_vlad.pkl"),
    "SIFT+BoF": ("sift_train_bof.pkl", "sift_test_bof.pkl"),
    "BRISK+VLAD": ("brisk_train_vlad.pkl", "brisk_test_vlad.pkl"),
    "BRISK+BoF": ("brisk_train_bof.pkl", "brisk_test_bof.pkl"),
}

results = {}

for name, (train_file, test_file) in combinations.items():
    print(f"\n正在处理 {name}...")
    sift_train_bof, sift_train_labels = load_bof_features(train_file)
    sift_test_bof, sift_test_labels = load_bof_features(test_file)

    retrieval_results, retrieval_time = knn_retrieval(sift_train_bof, sift_train_labels, sift_test_bof, sift_test_labels)
    recall, precision, mean_ap = compute_metrics(retrieval_results, sift_test_labels)

    results[name] = {
        "Recall": recall,
        "Precision": precision,
        "MAP": mean_ap,
        "Retrieval Time": retrieval_time,
    }

# 显示所有结果
for name, metrics in results.items():
    print(f"\n==== {name} ====")
    print(f"召回率: {metrics['Recall']:.4f}")
    print(f"精度: {metrics['Precision']:.4f}")
    print(f"MAP: {metrics['MAP']:.4f}")
    print(f"检索时间: {metrics['Retrieval Time']:.2f} 秒")
