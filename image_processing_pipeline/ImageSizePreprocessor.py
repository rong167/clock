import os
import cv2
import numpy as np
from tqdm import tqdm

class ImageSizePreprocessor:
    def __init__(self, input_folder, output_folder, target_size=128):
        """
        初始化影像處理器
        :param input_folder: 原始資料夾（包含多層資料夾）
        :param output_folder: 處理後輸出資料夾
        :param target_size: 最終輸出的影像大小（預設 128x128）
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_size = target_size
        self.image_paths = self._collect_images()

        # 創建輸出資料夾
        os.makedirs(self.output_folder, exist_ok=True)

    def _collect_images(self):
        """
        遞迴讀取所有子資料夾內的圖片，並保留相對路徑
        :return: 包含相對路徑的圖片路徑列表
        """
        image_paths = []
        for root, _, files in os.walk(self.input_folder):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    # 計算相對路徑
                    rel_path = os.path.relpath(os.path.join(root, file), self.input_folder)
                    image_paths.append(rel_path)
        return image_paths

    def _get_output_path(self, rel_path):
        """
        根據相對路徑生成對應的輸出路徑
        :param rel_path: 相對路徑
        :return: 輸出路徑
        """
        # 將相對路徑中的反斜線轉為正斜線（Windows相容性）
        rel_path = rel_path.replace('\\', '/')
        
        # 分離資料夾路徑和檔名
        dir_path, filename = os.path.split(rel_path)
        
        # 創建對應的子資料夾
        output_dir = os.path.join(self.output_folder, dir_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 確保輸出檔名使用 .jpg 副檔名
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}.jpg"
        
        return os.path.join(output_dir, output_filename)

    def _preprocess_image(self, image_path):
        """
        讀取並處理單張圖片（Padding + Resize）
        :param image_path: 圖片路徑
        :return: 處理後的影像 (numpy array) 或 None
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 讀取灰階
        if image is None:
            return None  # 避免讀取失敗時影響程序

        h, w = image.shape
        max_dim = max(h, w)

        # 計算 Padding 讓圖片變正方形
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left

        # 補白邊 (255 為白色)
        squared_img = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)

        # Resize 到目標大小
        resized_img = cv2.resize(squared_img, (self.target_size, self.target_size))

        return resized_img

    def process_all_images(self):
        """
        批量處理所有圖片並儲存，保持原始資料夾結構和檔名
        """
        successful_count = 0
        failed_count = 0
        
        for rel_path in tqdm(self.image_paths, desc="Processing Images"):
            full_input_path = os.path.join(self.input_folder, rel_path)
            processed_img = self._preprocess_image(full_input_path)
            
            if processed_img is not None:
                output_path = self._get_output_path(rel_path)
                success = cv2.imwrite(output_path, processed_img)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
                    print(f"儲存失敗: {output_path}")
            else:
                failed_count += 1
                print(f"處理失敗: {full_input_path}")
        
        print(f"完成！成功處理 {successful_count} 張，失敗 {failed_count} 張")
        print(f"輸出資料夾: {self.output_folder}")

# 執行影像處理
if __name__ == "__main__":
    # 設定輸入與輸出資料夾
    input_root_folder = "./box_detection_output"
    output_folder = "./resize"

    # 初始化並執行處理
    processor = ImageSizePreprocessor(input_root_folder, output_folder, target_size=256)
    processor.process_all_images()