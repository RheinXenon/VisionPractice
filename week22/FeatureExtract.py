import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 进度条库

# 设置图像数据集路径
image_folder = "../image"

# 目标保存路径
feature_save_path = "./features"
os.makedirs(feature_save_path, exist_ok=True)

# 定义特征提取方法
def extract_features(image_path, detector):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    keypoints, descriptors = detector.detectAndCompute(image, None)
    
    return descriptors

# 选择 SIFT 和 BRISK 作为特征提取器（不限制特征点数）
sift = cv2.SIFT_create()
brisk = cv2.BRISK_create()

# 遍历数据集
all_image_paths = []
all_labels = []
label_dict = {}

# 遍历所有车牌文件夹
for label, subdir in enumerate(os.listdir(image_folder)):
    subdir_path = os.path.join(image_folder, subdir)
    if os.path.isdir(subdir_path):
        label_dict[label] = subdir
        for filename in os.listdir(subdir_path):
            if filename.endswith(".jpg"):
                all_image_paths.append(os.path.join(subdir_path, filename))
                all_labels.append(label)

# 划分训练集和测试集
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_image_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# 处理并保存特征
def process_and_save_features(image_paths, labels, detector, feature_name):
    feature_dict = {}

    print(f"提取 {feature_name} 特征...")
    for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc=f"{feature_name} 进度"):
        descriptors = extract_features(img_path, detector)
        if descriptors is not None:
            if label not in feature_dict:
                feature_dict[label] = []
            feature_dict[label].append(descriptors)

    # 保存完整特征数据
    save_file = os.path.join(feature_save_path, f"{feature_name}_features.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(feature_dict, f)
    print(f"{feature_name} 特征已保存到: {save_file}")

# 提取和保存 SIFT 特征
process_and_save_features(train_paths, train_labels, sift, "sift_train")
process_and_save_features(test_paths, test_labels, sift, "sift_test")

# 提取和保存 BRISK 特征
process_and_save_features(train_paths, train_labels, brisk, "brisk_train")
process_and_save_features(test_paths, test_labels, brisk, "brisk_test")

# 保存数据划分信息
split_info = {
    "train_paths": train_paths,
    "test_paths": test_paths,
    "train_labels": train_labels,
    "test_labels": test_labels,
    "label_dict": label_dict
}
with open(os.path.join(feature_save_path, "data_split.pkl"), "wb") as f:
    pickle.dump(split_info, f)

print("数据划分信息已保存。")
