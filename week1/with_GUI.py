import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

class ImageMatcherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像匹配")
        
        # 按钮
        self.btn_select = tk.Button(root, text="选择查询图像", command=self.load_query_image)
        self.btn_select.pack()
        
        self.canvas1 = tk.Canvas(root, width=300, height=300)
        self.canvas1.pack()
        
        self.btn_match = tk.Button(root, text="开始匹配", command=self.find_best_match)
        self.btn_match.pack()
        
        self.canvas2 = tk.Canvas(root, width=300, height=300)
        self.canvas2.pack()
        
        self.label_result = tk.Label(root, text="")
        self.label_result.pack()
        
        self.query_img = None
        self.match_result_img = None

        # 设定待匹配图片的文件夹
        self.dataset_dir = "test_dataset"  # 指定数据集文件夹
        self.dataset = self.load_dataset_images()  # 获取所有符合要求的图片路径

    def load_dataset_images(self):
        """ 获取 test_dataset 文件夹下所有 jpg 和 png 图片路径 """
        dataset_images = []
        if os.path.exists(self.dataset_dir):
            for file in os.listdir(self.dataset_dir):
                if file.lower().endswith((".jpg", ".png")):
                    dataset_images.append(os.path.join(self.dataset_dir, file))
        return dataset_images

    def load_query_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.query_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(file_path, self.canvas1)

    def find_best_match(self):
        if self.query_img is None:
            self.label_result.config(text="请先选择查询图像！")
            return

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.query_img, None)
        bf = cv2.BFMatcher()

        best_match = 0
        best_img_path = None
        score_list = []  # 存储所有匹配分数
        start_time = time.time()

        for img_path in self.dataset:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            kp2, des2 = sift.detectAndCompute(img, None)
            if des2 is None or des1 is None:
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            
            score = len(good_matches)
            score_list.append((img_path, score))  # 记录图像路径和匹配分数

            if score > best_match:
                best_match = score
                best_img_path = img_path

        match_time = time.time() - start_time

        # 显示最佳匹配的图片
        if best_img_path:
            self.display_image(best_img_path, self.canvas2)

        # 按匹配分数排序
        score_list.sort(key=lambda x: x[1], reverse=True)
        top_scores = score_list[:6]

        # 显示匹配分数列表
        result_text = f"最佳匹配: {best_img_path}\n匹配时间: {match_time:.4f}s\n\n相似度排名:\n"
        for img_path, score in top_scores:
            result_text += f"{img_path}: {score} 个匹配点\n"

        self.label_result.config(text=result_text)

    def display_image(self, path, canvas):
        img = Image.open(path)
        img = img.resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        canvas.create_image(150, 150, image=img)
        canvas.image = img

root = tk.Tk()
app = ImageMatcherGUI(root)
root.mainloop()
