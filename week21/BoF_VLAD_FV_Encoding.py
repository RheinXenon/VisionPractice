import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

# 设定参数
num_clusters = 100  # 视觉词汇表大小
feature_save_path = "./features"

# 加载数据
def load_features(feature_file):
    with open(feature_file, "rb") as f:
        return pickle.load(f)

sift_train_features = load_features(os.path.join(feature_save_path, "sift_train_features.pkl"))
brisk_train_features = load_features(os.path.join(feature_save_path, "brisk_train_features.pkl"))
sift_test_features = load_features(os.path.join(feature_save_path, "sift_test_features.pkl"))
brisk_test_features = load_features(os.path.join(feature_save_path, "brisk_test_features.pkl"))

# 加载视觉词典
with open(os.path.join(feature_save_path, "sift_codebook.pkl"), "rb") as f:
    sift_kmeans = pickle.load(f)
with open(os.path.join(feature_save_path, "brisk_codebook.pkl"), "rb") as f:
    brisk_kmeans = pickle.load(f)

# 加载 Fisher Vector GMM 模型（如果不存在，则训练）
gmm_path_sift = os.path.join(feature_save_path, "sift_gmm.pkl")
gmm_path_brisk = os.path.join(feature_save_path, "brisk_gmm.pkl")

if os.path.exists(gmm_path_sift):
    with open(gmm_path_sift, "rb") as f:
        sift_gmm = pickle.load(f)
else:
    sift_descriptors = np.vstack([desc for desc_list in sift_train_features.values() for desc in desc_list if desc is not None])
    sift_gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=42).fit(sift_descriptors)
    with open(gmm_path_sift, "wb") as f:
        pickle.dump(sift_gmm, f)

if os.path.exists(gmm_path_brisk):
    with open(gmm_path_brisk, "rb") as f:
        brisk_gmm = pickle.load(f)
else:
    brisk_descriptors = np.vstack([desc for desc_list in brisk_train_features.values() for desc in desc_list if desc is not None])
    brisk_gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', random_state=42).fit(brisk_descriptors)
    with open(gmm_path_brisk, "wb") as f:
        pickle.dump(brisk_gmm, f)


# 计算 BoF 直方图
def compute_bof_histograms(feature_dict, kmeans_model):
    bof_histograms = []
    image_labels = []
    for label, desc_list in tqdm(feature_dict.items(), desc="计算 BoF 直方图"):
        for descriptors in desc_list:
            if descriptors is not None and len(descriptors) > 0:
                words = kmeans_model.predict(descriptors)
                hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
                hist = hist.astype(float) / hist.sum()  # 归一化
                bof_histograms.append(hist)
                image_labels.append(label)
    return np.array(bof_histograms), np.array(image_labels)


# 计算 VLAD 编码
def compute_vlad(feature_dict, kmeans_model):
    vlad_features = []
    image_labels = []
    cluster_centers = kmeans_model.cluster_centers_

    for label, desc_list in tqdm(feature_dict.items(), desc="计算 VLAD 编码"):
        for descriptors in desc_list:
            if descriptors is not None and len(descriptors) > 0:
                words = kmeans_model.predict(descriptors)
                residuals = np.zeros((num_clusters, descriptors.shape[1]))

                for i, word in enumerate(words):
                    residuals[word] += descriptors[i] - cluster_centers[word]

                residuals = residuals.flatten()
                residuals /= np.linalg.norm(residuals, ord=2)  # 归一化
                vlad_features.append(residuals)
                image_labels.append(label)

    return np.array(vlad_features), np.array(image_labels)


# 计算 Fisher Vector (FV) 编码
def compute_fisher_vector(feature_dict, gmm_model):
    fv_features = []
    image_labels = []

    for label, desc_list in tqdm(feature_dict.items(), desc="计算 Fisher Vector 编码"):
        for descriptors in desc_list:
            if descriptors is not None and len(descriptors) > 0:
                probs = gmm_model.predict_proba(descriptors)
                means = gmm_model.means_
                covs = gmm_model.covariances_

                fisher_vector = np.hstack([
                    np.sum(probs[:, i, None] * (descriptors - means[i]) / np.sqrt(covs[i]), axis=0)
                    for i in range(num_clusters)
                ])
                fisher_vector /= np.linalg.norm(fisher_vector, ord=2)  # 归一化
                fv_features.append(fisher_vector)
                image_labels.append(label)

    return np.array(fv_features), np.array(image_labels)


# 计算并保存编码
def process_and_save(feature_dict, kmeans_model, gmm_model, prefix):
    print(f"计算 {prefix} BoF 直方图...")
    bof_features, bof_labels = compute_bof_histograms(feature_dict, kmeans_model)
    with open(os.path.join(feature_save_path, f"{prefix}_bof.pkl"), "wb") as f:
        pickle.dump((bof_features, bof_labels), f)

    print(f"计算 {prefix} VLAD 编码...")
    vlad_features, vlad_labels = compute_vlad(feature_dict, kmeans_model)
    with open(os.path.join(feature_save_path, f"{prefix}_vlad.pkl"), "wb") as f:
        pickle.dump((vlad_features, vlad_labels), f)

    print(f"计算 {prefix} Fisher Vector 编码...")
    fv_features, fv_labels = compute_fisher_vector(feature_dict, gmm_model)
    with open(os.path.join(feature_save_path, f"{prefix}_fv.pkl"), "wb") as f:
        pickle.dump((fv_features, fv_labels), f)


# 处理训练集
process_and_save(sift_train_features, sift_kmeans, sift_gmm, "sift_train")
process_and_save(brisk_train_features, brisk_kmeans, brisk_gmm, "brisk_train")

# 处理测试集
process_and_save(sift_test_features, sift_kmeans, sift_gmm, "sift_test")
process_and_save(brisk_test_features, brisk_kmeans, brisk_gmm, "brisk_test")

print("所有 BoF, VLAD, FV 编码已保存！")
