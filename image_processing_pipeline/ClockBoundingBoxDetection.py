import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # 記得放在最上面

class ClockBoundingBoxDetection:
    def __init__(self, input_root, output_root):
        self.input_root = input_root
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        # 固定裁切範圍（left, upper, right, lower）
        self.crop_box = (0, 50, 960, 780)  # 可根據你的資料自訂範圍

    def detect_and_crop_bounding_box(self, input_path, output_path):
        preprocess_image = Image.open(input_path)

        # 固定裁切範圍
        preprocess_image = preprocess_image.crop(self.crop_box)

        # 顯示裁切後的圖片給使用者確認
        #plt.imshow(preprocess_image)
        #plt.title("Preview of Cropped Image")
        #plt.axis('off')
        #plt.show()  # 顯示完後可按右上角 X 關閉視窗

        # 確保圖片為 RGB 或灰階
        if preprocess_image.mode not in ("RGB", "L"):
            preprocess_image = preprocess_image.convert("RGB")

        # 轉換為灰階處理
        cv2_preprocess_image = cv2.cvtColor(np.asarray(preprocess_image), cv2.COLOR_RGB2BGR)
        cv2_preprocess_image = cv2.cvtColor(cv2_preprocess_image, cv2.COLOR_BGR2GRAY)

        cv2_preprocess_image = cv2.medianBlur(cv2_preprocess_image, ksize=5) 
        cv2_preprocess_image = cv2.medianBlur(cv2_preprocess_image, ksize=5) 
        cv2_preprocess_image = cv2.medianBlur(cv2_preprocess_image, ksize=5) 
        cv2_preprocess_image = cv2.medianBlur(cv2_preprocess_image, ksize=5) 

        # 模糊與邊緣偵測
        blurred = cv2.GaussianBlur(cv2_preprocess_image, (15, 15), 0)
        binaryIMG = cv2.Canny(blurred, 20, 160)

        # 定義 kernel

        kernel = np.ones((7, 7), np.uint8)



        # 再做閉運算（填補內部空洞）
        binaryIMG = cv2.morphologyEx(binaryIMG, cv2.MORPH_CLOSE, kernel, iterations=10)

        # 找輪廓
        cnts, _ = cv2.findContours(binaryIMG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找最大輪廓
        max_area = 0
        max_cnts = None
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h > max_area:
                max_area = w * h
                max_cnts = c

        if max_cnts is not None:
            (x, y, w, h) = cv2.boundingRect(max_cnts)
            cropped_image = preprocess_image.crop((x, y, x + w, y + h))
            cropped_image.save(output_path)
            print(f"已儲存裁剪圖片: {output_path}")
        else:
            print(f"找不到時鐘輪廓: {input_path}")

    def process_folders(self):
        for subdir, _, files in os.walk(self.input_root):
            if not files:
                continue

            relative_path = os.path.relpath(subdir, self.input_root)
            output_folder = os.path.join(self.output_root, relative_path)
            os.makedirs(output_folder, exist_ok=True)

            for filename in files:
                if filename.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
                    input_path = os.path.join(subdir, filename)
                    output_filename = os.path.splitext(filename)[0] + ".jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    self.detect_and_crop_bounding_box(input_path, output_path)

if __name__ == '__main__':
    input_root = './image_preprocessor_MCI'  # 你的輸入根目錄
    output_root = './box_detection_MCI'  # 你的輸出根目錄

    clock_detector = ClockBoundingBoxDetection(input_root, output_root)
    clock_detector.process_folders()
