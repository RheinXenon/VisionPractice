import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# 设定参数
num_clusters = 500  # GMM 组件数（与 KMeans 相同）
feature_save_path = "./features"

# 加载特征数据
def load_features(feature_file):
    print(f"正在加载特征文件: {feature_file} ...")
    with open(feature_file, "rb") as f:
        return pickle.load(f)

sift_train_features = load_features(os.path.join(feature_save_path, "sift_train_features.pkl"))
brisk_train_features = load_features(os.path.join(feature_save_path, "brisk_train_features.pkl"))

# 提取所有 SIFT 和 BRISK 特征
def collect_descriptors(feature_dict):
    descriptors_list = []
    for desc_list in tqdm(feature_dict.values(), desc="收集特征"):
        for descriptors in desc_list:
            if descriptors is not None and len(descriptors) > 0:
                descriptors_list.append(descriptors)
    return np.vstack(descriptors_list) if descriptors_list else None

sift_descriptors = collect_descriptors(sift_train_features)
brisk_descriptors = collect_descriptors(brisk_train_features)

# 训练 GMM
def train_gmm(descriptors, num_clusters, save_path):
    print(f"开始训练 GMM ({num_clusters} clusters)...")
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', verbose=1)
    gmm.fit(descriptors)
    with open(save_path, "wb") as f:
        pickle.dump(gmm, f)
    print(f"GMM 模型已保存至 {save_path}")

# 训练并保存 GMM
if sift_descriptors is not None:
    train_gmm(sift_descriptors, num_clusters, os.path.join(feature_save_path, "sift_gmm.pkl"))

if brisk_descriptors is not None:
    train_gmm(brisk_descriptors, num_clusters, os.path.join(feature_save_path, "brisk_gmm.pkl"))
