import os
import pickle
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm

# 设定参数
num_clusters = 100  # 视觉词汇表大小
feature_save_path = "./features"

# 加载数据划分信息
with open(os.path.join(feature_save_path, "data_split.pkl"), "rb") as f:
    data_split = pickle.load(f)

# 加载训练集特征
def load_features(feature_file):
    with open(feature_file, "rb") as f:
        return pickle.load(f)

sift_train_features = load_features(os.path.join(feature_save_path, "sift_train_features.pkl"))
brisk_train_features = load_features(os.path.join(feature_save_path, "brisk_train_features.pkl"))

# 组合所有特征用于训练码本
def stack_features(feature_dict):
    all_descriptors = []
    for desc_list in feature_dict.values():
        for descriptors in desc_list:
            all_descriptors.append(descriptors)
    return np.vstack(all_descriptors)

print("构建视觉词汇表...")
sift_descriptors = stack_features(sift_train_features)
brisk_descriptors = stack_features(brisk_train_features)

# KMeans 训练视觉词典（码本）
print("训练 SIFT 视觉词典...")
sift_kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
sift_kmeans.fit(sift_descriptors)

print("训练 BRISK 视觉词典...")
brisk_kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
brisk_kmeans.fit(brisk_descriptors)

# 保存码本
with open(os.path.join(feature_save_path, "sift_codebook.pkl"), "wb") as f:
    pickle.dump(sift_kmeans, f)
with open(os.path.join(feature_save_path, "brisk_codebook.pkl"), "wb") as f:
    pickle.dump(brisk_kmeans, f)

print("视觉词典已保存。")

# # 计算 BoF 直方图
# def compute_bof_histograms(feature_dict, kmeans_model):
#     bof_histograms = []
#     image_labels = []
#     for label, desc_list in tqdm(feature_dict.items(), desc="计算 BoF 直方图"):
#         for descriptors in desc_list:
#             if descriptors is not None and len(descriptors) > 0:
#                 words = kmeans_model.predict(descriptors)
#                 hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
#                 hist = hist.astype(float) / hist.sum()  # 归一化
#                 bof_histograms.append(hist)
#                 image_labels.append(label)
#     return np.array(bof_histograms), np.array(image_labels)

# print("计算 SIFT BoF 直方图...")
# sift_train_bof, sift_train_labels = compute_bof_histograms(sift_train_features, sift_kmeans)
# print("计算 BRISK BoF 直方图...")
# brisk_train_bof, brisk_train_labels = compute_bof_histograms(brisk_train_features, brisk_kmeans)

# # 保存 BoF 直方图
# with open(os.path.join(feature_save_path, "sift_train_bof.pkl"), "wb") as f:
#     pickle.dump((sift_train_bof, sift_train_labels), f)
# with open(os.path.join(feature_save_path, "brisk_train_bof.pkl"), "wb") as f:
#     pickle.dump((brisk_train_bof, brisk_train_labels), f)

# print("BoF 直方图已保存。")

# # 计算测试集的 BoF
# sift_test_features = load_features(os.path.join(feature_save_path, "sift_test_features.pkl"))
# brisk_test_features = load_features(os.path.join(feature_save_path, "brisk_test_features.pkl"))

# print("计算 SIFT 测试集 BoF 直方图...")
# sift_test_bof, sift_test_labels = compute_bof_histograms(sift_test_features, sift_kmeans)
# print("计算 BRISK 测试集 BoF 直方图...")
# brisk_test_bof, brisk_test_labels = compute_bof_histograms(brisk_test_features, brisk_kmeans)

# # 保存测试集 BoF
# with open(os.path.join(feature_save_path, "sift_test_bof.pkl"), "wb") as f:
#     pickle.dump((sift_test_bof, sift_test_labels), f)
# with open(os.path.join(feature_save_path, "brisk_test_bof.pkl"), "wb") as f:
#     pickle.dump((brisk_test_bof, brisk_test_labels), f)

# print("测试集 BoF 直方图已保存。")
