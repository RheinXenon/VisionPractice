import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)  # 目标图像
img2 = cv2.imread("match.png", cv2.IMREAD_GRAYSCALE)  # 待匹配图像

# 1. SIFT 特征匹配
sift = cv2.SIFT_create()
start_time = time.time()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches_sift = bf.knnMatch(des1, des2, k=2)
good_sift = [m for m, n in matches_sift if m.distance < 0.75 * n.distance]
sift_time = time.time() - start_time

# 2. ORB 特征匹配
orb = cv2.ORB_create()
start_time = time.time()
kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf_orb.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
orb_time = time.time() - start_time

# 绘制匹配结果
img_sift = cv2.drawMatches(img1, kp1, img2, kp2, good_sift[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb, matches_orb[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_sift)
plt.title(f"SIFT Matching (Time: {sift_time:.4f}s)")
plt.subplot(1, 2, 2)
plt.imshow(img_orb)
plt.title(f"ORB Matching (Time: {orb_time:.4f}s)")
plt.show()
