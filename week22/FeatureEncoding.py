import os
import pickle
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# 设定参数
num_clusters = 500  # 视觉词汇表大小
batch_size = 1000   # MiniBatchKMeans 的批量大小
num_iterations = 2000  # 迭代次数
sample_size = 10000  # 每次迭代的样本数
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

# 训练 MiniBatchKMeans 视觉词典（码本）
print("训练 SIFT 视觉词典...")
sift_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=batch_size, n_init=10)
for _ in tqdm(range(num_iterations)):
    sample_indices = np.random.choice(len(sift_descriptors), size=sample_size, replace=False)
    sift_kmeans.partial_fit(sift_descriptors[sample_indices])

print("训练 BRISK 视觉词典...")
brisk_kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=batch_size, n_init=10)
for _ in tqdm(range(num_iterations)):
    sample_indices = np.random.choice(len(brisk_descriptors), size=sample_size, replace=False)
    brisk_kmeans.partial_fit(brisk_descriptors[sample_indices])

# 保存码本
with open(os.path.join(feature_save_path, "sift_codebook.pkl"), "wb") as f:
    pickle.dump(sift_kmeans, f)
with open(os.path.join(feature_save_path, "brisk_codebook.pkl"), "wb") as f:
    pickle.dump(brisk_kmeans, f)

print("视觉词典已保存。")
