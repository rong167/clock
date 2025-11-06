import matplotlib.pyplot as plt
from PIL import Image, ExifTags
import cv2 
import numpy as np
import os

class ImagePreprocessor:
    def __init__(self):
        self.processed_image = None

    def _rotate_image(self, image):
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(image._getexif().items())

            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        return image

    def _resize_image(self, image):
        if image.size[0] > image.size[1]:
            image = image.resize((int(image.size[0] * 972 / image.size[1]), 972), Image.BICUBIC)
        else:
            image = image.resize((972, int(image.size[1] * 972 / image.size[0])), Image.BICUBIC)
        return image

    def _apply_filters(self, image, balance):
        if balance:
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel1)
            div = np.float32(image) / (close)
            image = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        return image

    def _binary_image(self, cv2_image):
        hist = cv2.calcHist([cv2_image], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        Q = hist_norm.cumsum()
        bins = np.arange(256)
        fn_min = np.inf
        thresh1 = 0
        thresh2 = 0

        for i in range(1, 256):
            for j in range(i + 1, 256):
                p1, p2, p3 = np.hsplit(hist_norm, [i, j])
                q1, q2, q3 = Q[i], Q[j] - Q[i], Q[255] - Q[j]
                if q1 < 1.e-6 or q2 < 1.e-6 or q3 < 1.e-6:
                    continue
                b1, b2, b3 = np.hsplit(bins, [i, j])
                m1, m2, m3 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2, np.sum(p3 * b3) / q3
                v1 = np.sum(((b1 - m1) ** 2) * p1) / q1
                v2 = np.sum(((b2 - m2) ** 2) * p2) / q2
                v3 = np.sum(((b3 - m3) ** 2) * p3) / q3
                fn = v1 * q1 + v2 * q2 + v3 * q3
                if fn < fn_min:
                    fn_min = fn
                    thresh1 = i
                    thresh2 = j

        _, binary_image = cv2.threshold(cv2_image, thresh2*1.03, 255, cv2.THRESH_BINARY)
        return binary_image

    def preprocess_image(self, image_path, balance=True):
        image = Image.open(image_path)
        image = self._rotate_image(image)
        image = self._resize_image(image)
        # 固定裁切範圍（left, upper, right, lower）
        self.crop_box = (0, 0, 971, 1300)  # 可根據你的資料自訂範圍
        image = image.crop(self.crop_box)

        # 顯示裁切後的圖片給使用者確認
        #plt.imshow(image)
        #plt.title("Preview of Cropped Image")
        #plt.axis('off')
        #plt.show()  # 顯示完後可按右上角 X 關閉視窗

        try:
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            blur_image = cv2.GaussianBlur(cv2_image, (5, 5), 0)
        except:
            image = image.convert('L')
            cv2_image = np.array(image)
            blur_image = cv2.GaussianBlur(cv2_image, (5, 5), 0)

        cv2_image = self._apply_filters(blur_image, balance)
        binary_image = self._binary_image(cv2_image)

        self.processed_image = binary_image

    def save_processed_image(self, output_path):
        if self.processed_image is not None:
            processed_color = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
            success, encoded_img = cv2.imencode('.jpg', processed_color)
            if success:
                encoded_img.tofile(output_path)
                print(f"已儲存：{output_path}")
            else:
                print(f"❗ 無法儲存：{output_path}")

if __name__ == '__main__':
    input_folder = "../資料集/監督訓練資料/image/前處理前/11/"
    output_folder = "./image_preprocessor"
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    preprocessor = ImagePreprocessor()

    for filename in image_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            preprocessor.preprocess_image(input_path)
            preprocessor.save_processed_image(output_path)
        except Exception as e:
            print(f"處理失敗：{filename}，錯誤：{e}")