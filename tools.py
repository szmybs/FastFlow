from PIL import Image
import os
import numpy as np
from pathlib import Path

def is_close_to_white(rgb, threshold=50):
    """
    判断一个像素是否接近白色
    使用欧几里得距离来衡量和白色的距离，如果距离小于阈值，则认为接近白色
    :param rgb: 像素的RGB值
    :param threshold: 阈值，决定"接近白色"的标准，默认50
    :return: 如果接近白色返回True，否则返回False
    """
    # 计算与白色的欧几里得距离
    distance = np.sqrt((255 - rgb[0])**2 + (255 - rgb[1])**2 + (255 - rgb[2])**2)
    return distance < threshold

def process_image(image_path, threshold=50):
    # 使用PIL打开图像
    img = Image.open(image_path)
    
    # 转换为RGB模式（确保图像是RGB格式）
    img_rgb = img.convert('RGB')

    # 获取图像数据
    img_data = np.array(img_rgb)

    # 创建一个黑色图像，尺寸与原图相同
    processed_img_data = np.zeros_like(img_data)

    # 遍历所有像素，将接近白色的像素设为白色，其它设为黑色
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):
            if is_close_to_white(img_data[i, j], threshold):
                processed_img_data[i, j] = [255, 255, 255]  # 设置为白色
            else:
                processed_img_data[i, j] = [0, 0, 0]  # 设置为黑色

    # 将处理后的数据转换为PIL图像
    processed_img = Image.fromarray(processed_img_data)

    # 转换为灰度图
    gray_img = processed_img.convert('L')

    return gray_img

def save_image(image, save_path):
    # 保存图像
    image.save(save_path)

def process_folder(folder_path, threshold=50):
    # 遍历文件夹及其子文件夹中的所有图片
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取图片文件的完整路径
            file_path = os.path.join(root, file)

            # 仅处理图像文件（例如png, jpg, jpeg等）
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
                # 处理图片
                processed_img = process_image(file_path, threshold)

                # 修改文件名为"原文件名_mask"
                file_name, file_ext = os.path.splitext(file)
                new_file_name = f"{file_name}_mask{file_ext}"

                # 保存处理后的图片
                new_file_path = os.path.join(root, new_file_name)
                save_image(processed_img, new_file_path)
                print(f"处理并保存图片: {new_file_path}")

# 主程序入口
if __name__ == "__main__":
    folder_path = '你的文件夹路径'  # 请替换为你的文件夹路径
    process_folder("./mvtec-ad/plate/ground_truth/stains", threshold=25)

