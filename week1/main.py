import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def convert_to_jpg_with_white_bg(image_path):
    # 读取图像，包括带透明通道的PNG
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 检查是否具有透明通道 (4通道)
    if image.shape[2] == 4:
        # 分离通道
        b, g, r, alpha = cv2.split(image)
        
        # 创建一个背景
        white_background = np.ones_like(alpha) * 0
        
        # 使用alpha通道将前景和背景混合
        alpha = alpha / 255.0
        b = cv2.convertScaleAbs(b * alpha + white_background * (1 - alpha))
        g = cv2.convertScaleAbs(g * alpha + white_background * (1 - alpha))
        r = cv2.convertScaleAbs(r * alpha + white_background * (1 - alpha))
        
        # 合并通道
        image = cv2.merge([b, g, r])
    else:
        # 如果没有透明通道，则直接转为3通道（如果是灰度图像）
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 转为灰度图像（SIFT等算法需要灰度图像）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image

# 读取图像并转换为JPG格式，处理透明背景

img1 = convert_to_jpg_with_white_bg("./match_pairs/input1.png")  # 目标图像
img2 = convert_to_jpg_with_white_bg("./match_pairs/match1.png")  # 待匹配图像

# 调整图像尺寸
height, width = img1.shape
img2 = cv2.resize(img2, (width, height))

# SIFT 特征匹配
sift = cv2.SIFT_create()
start_time = time.time()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches_sift = bf.knnMatch(des1, des2, k=2)
good_sift = [m for m, n in matches_sift if m.distance < 0.75 * n.distance]
sift_time = time.time() - start_time
sift_matches_count = len(good_sift)

# SURF 特征匹配 实际运行中发现SURF算法有专利 高版本的OpenCV并没有配置无法运行
# surf = cv2.xfeatures2d.SURF_create(400)
# start_time = time.time()
# kp1_surf, des1_surf = surf.detectAndCompute(img1, None)
# kp2_surf, des2_surf = surf.detectAndCompute(img2, None)
# bf_surf = cv2.BFMatcher()
# matches_surf = bf_surf.knnMatch(des1_surf, des2_surf, k=2)
# good_surf = [m for m, n in matches_surf if m.distance < 0.75 * n.distance]
# surf_time = time.time() - start_time

# ORB 特征匹配
orb = cv2.ORB_create()
start_time = time.time()
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
orb_time = time.time() - start_time
orb_matches_count = len(matches_orb)

# AKAZE 特征匹配
akaze = cv2.AKAZE_create()
start_time = time.time()
kp1_akaze, des1_akaze = akaze.detectAndCompute(img1, None)
kp2_akaze, des2_akaze = akaze.detectAndCompute(img2, None)
bf_akaze = cv2.BFMatcher(cv2.NORM_HAMMING)
matches_akaze = bf_akaze.knnMatch(des1_akaze, des2_akaze, k=2)
good_akaze = [m for m, n in matches_akaze if m.distance < 0.75 * n.distance]
akaze_time = time.time() - start_time
akaze_matches_count = len(good_akaze)

# BRISK 特征匹配
brisk = cv2.BRISK_create()
start_time = time.time()
kp1_brisk, des1_brisk = brisk.detectAndCompute(img1, None)
kp2_brisk, des2_brisk = brisk.detectAndCompute(img2, None)
bf_brisk = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_brisk = bf_brisk.match(des1_brisk, des2_brisk)
matches_brisk = sorted(matches_brisk, key=lambda x: x.distance)
brisk_time = time.time() - start_time
brisk_matches_count = len(matches_brisk)

# 绘制匹配结果
img_sift = cv2.drawMatches(img1, kp1, img2, kp2, good_sift[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_akaze = cv2.drawMatches(img1, kp1_akaze, img2, kp2_akaze, good_akaze[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_brisk = cv2.drawMatches(img1, kp1_brisk, img2, kp2_brisk, matches_brisk[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(img_sift)
plt.title(f"SIFT Matching\nTime: {sift_time:.4f}s, Matches: {sift_matches_count}")

plt.subplot(2, 2, 2)
plt.imshow(img_orb)
plt.title(f"ORB Matching\nTime: {orb_time:.4f}s, Matches: {orb_matches_count}")

plt.subplot(2, 2, 3)
plt.imshow(img_akaze)
plt.title(f"AKAZE Matching\nTime: {akaze_time:.4f}s, Matches: {akaze_matches_count}")

plt.subplot(2, 2, 4)
plt.imshow(img_brisk)
plt.title(f"BRISK Matching\nTime: {brisk_time:.4f}s, Matches: {brisk_matches_count}")

plt.tight_layout()
plt.show()
