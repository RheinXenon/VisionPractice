import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm

# 设定参数
num_clusters = 500  # 视觉词汇表大小
feature_save_path = "./features"

# 加载数据
def load_features(feature_file):
    print(f"正在加载特征文件: {feature_file} ...")
    with open(feature_file, "rb") as f:
        return pickle.load(f)

sift_train_features = load_features(os.path.join(feature_save_path, "sift_train_features.pkl"))
brisk_train_features = load_features(os.path.join(feature_save_path, "brisk_train_features.pkl"))
sift_test_features = load_features(os.path.join(feature_save_path, "sift_test_features.pkl"))
brisk_test_features = load_features(os.path.join(feature_save_path, "brisk_test_features.pkl"))

# 加载视觉词典
print("正在加载视觉词典...")
with open(os.path.join(feature_save_path, "sift_codebook.pkl"), "rb") as f:
    sift_kmeans = pickle.load(f)
with open(os.path.join(feature_save_path, "brisk_codebook.pkl"), "rb") as f:
    brisk_kmeans = pickle.load(f)

# 计算 BoF 直方图
def compute_bof_histograms(feature_dict, kmeans_model):
    print("开始计算 BoF 直方图...")
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
    print("BoF 直方图计算完成！")
    return np.array(bof_histograms), np.array(image_labels)

# 计算 VLAD 编码
def compute_vlad(feature_dict, kmeans_model):
    print("开始计算 VLAD 编码...")
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

    print("VLAD 编码计算完成！")
    return np.array(vlad_features), np.array(image_labels)

# 计算并保存编码
def process_and_save(feature_dict, kmeans_model, prefix):
    print(f"开始处理 {prefix} 特征数据...")

    print(f"计算 {prefix} BoF 直方图...")
    bof_features, bof_labels = compute_bof_histograms(feature_dict, kmeans_model)
    with open(os.path.join(feature_save_path, f"{prefix}_bof.pkl"), "wb") as f:
        pickle.dump((bof_features, bof_labels), f)
    print(f"{prefix} BoF 直方图已保存！")

    print(f"计算 {prefix} VLAD 编码...")
    vlad_features, vlad_labels = compute_vlad(feature_dict, kmeans_model)
    with open(os.path.join(feature_save_path, f"{prefix}_vlad.pkl"), "wb") as f:
        pickle.dump((vlad_features, vlad_labels), f)
    print(f"{prefix} VLAD 编码已保存！")

    print(f"{prefix} 特征处理完成！\n")

# 处理训练集
process_and_save(sift_train_features, sift_kmeans, "sift_train")
process_and_save(brisk_train_features, brisk_kmeans, "brisk_train")

# 处理测试集
process_and_save(sift_test_features, sift_kmeans, "sift_test")
process_and_save(brisk_test_features, brisk_kmeans, "brisk_test")

print("所有 BoF 和 VLAD 编码已完成并保存！")
